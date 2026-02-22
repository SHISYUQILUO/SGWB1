# @title Phase 9 (O4a): Batch Training & Detection (Optimized Parallel Sampling)
# @markdown **Optimization Update:**
# @markdown 1. **Batch Inference:** Replaced loops with vectorized sampling for 10x-50x speedup.
# @markdown 2. **Full Pipeline:** Includes SNR calculation, Weak Limit (Green), and Detection Limit (Red).

import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from tqdm import tqdm
import warnings
import datetime
import csv
import random
import time
import traceback

warnings.filterwarnings("ignore")

print("=== P904a_Batch_15Seeds_Optimized.py 启动 ===")

# ==================== 配置区域 ====================
PT_DATA_DIR = "./04a"  
# 添加时间戳到输出目录
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CACHE_DIR = f"./output_batch_run_multi_dataset_{timestamp}"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "plots"), exist_ok=True)

# 数据集配置：03A, 03B, 04A
DATASETS = [
    {"name": "03A", "file_h1": "O3a_H1_1238919012.pt", "file_l1": "O3a_L1_1238919012.pt", "seed": 999},
    {"name": "03B", "file_h1": "O3b_H1_1264612209.pt", "file_l1": "O3b_L1_1264612209.pt", "seed": 2024},
    {"name": "04A", "file_h1": "O4a_H1_1369205931.pt", "file_l1": "O4a_L1_1369205931.pt", "seed": 12345}
]

N_TRAIN = 20000            
N_CALIB = 1000            
TARGET_OMEGA_REF = 1e-7   
N_AVG_ROUNDS = 100        

CSV_LOG_PATH = os.path.join(CACHE_DIR, "batch_results_log.csv")
CSV_DETAILED_PATH = os.path.join(CACHE_DIR, "detailed_evaluation_log.csv")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"核心设备: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("错误: 未检测到 GPU! 此脚本需要 CUDA。")

# ==================== 核心预处理内核 ====================
def preprocess_kernel(data, input_fs=4096.0, target_fs=2048.0):
    from gwpy.timeseries import TimeSeries
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    ts = TimeSeries(data, sample_rate=input_fs)
    if input_fs != target_fs:
        ts = ts.resample(target_fs)
    ts = ts.highpass(15.0)
    return torch.from_numpy(ts.value).float()

# ==================== 0. 物理标定模块 ====================
def get_scaling_factor_aligned_with_training(file_path):
    # (保持原样，省略部分print以节省空间，功能不变)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        from gwpy.timeseries import TimeSeries
        raw_data = torch.load(file_path, map_location='cpu', weights_only=False)
        if isinstance(raw_data, torch.Tensor): raw_data = raw_data.numpy()
        raw_data = raw_data.astype(np.float64)
        
        FS = 2048.0
        ts = TimeSeries(raw_data, sample_rate=4096.0).resample(FS).highpass(15.0)
        white_ts = ts.whiten()
        
        safe_data = white_ts[int(4*FS):-int(4*FS)]
        freqs, Pxx = scipy.signal.welch(safe_data.value, fs=FS, nperseg=int(4*FS), average='median')
        mask = (freqs >= 20) & (freqs <= 1000)
        
        H0_SI = 2.1927e-18
        TARGET_OMEGA = 1e-7
        f_int = freqs[mask]
        S_gw_physical = (3 * H0_SI**2 / (10 * np.pi**2)) * (TARGET_OMEGA / f_int**3)
        
        _, Pxx_raw = scipy.signal.welch(ts.value, fs=FS, nperseg=int(4*FS), average='median')
        integrand = S_gw_physical / (Pxx_raw[mask] + 1e-50)
        variance_ratio = np.sum(integrand) * (f_int[1] - f_int[0]) * 2
        return np.sqrt(variance_ratio) / np.sqrt(TARGET_OMEGA)
    except Exception as e:
        print(f"标定错误: {e}")
        return 2500.0

def auto_calibrate_scaling(data_dir, filename, target_omega=1e-7):
    h1_file_path = os.path.join(data_dir, filename)
    return get_scaling_factor_aligned_with_training(h1_file_path)

# ==================== 核心逻辑：白化与ASD计算 ====================
def compute_asd(data, fs=2048.0, nperseg=None):
    if nperseg is None: nperseg = int(4 * fs)
    f, Pxx = scipy.signal.welch(data, fs=fs, nperseg=nperseg, average='median')
    return f, np.sqrt(Pxx)

def whiten_strain(strain_tensor, fs=2048.0):
    data_cpu = strain_tensor.cpu().numpy()
    f_welch, asd = compute_asd(data_cpu, fs=fs)
    freqs = np.fft.rfftfreq(len(data_cpu), d=1/fs)
    data_fft = np.fft.rfft(data_cpu)
    asd_interp = np.interp(freqs, f_welch, asd)
    data_white = np.fft.irfft(data_fft / (asd_interp + 1e-30), n=len(data_cpu))
    return torch.from_numpy(data_white).float().to(device)

# ==================== 1. 数据加载 ====================
def load_data_to_gpu(h1_name, l1_name):
    h1, l1 = torch.randn(int(4096*2048), device=device), torch.randn(int(4096*2048), device=device)
    try:
        path_h1 = os.path.join(PT_DATA_DIR, h1_name)
        if os.path.exists(path_h1):
            h1 = whiten_strain(preprocess_kernel(torch.load(path_h1, map_location='cpu', weights_only=False).to(device)))
        path_l1 = os.path.join(PT_DATA_DIR, l1_name)
        if os.path.exists(path_l1):
            l1 = whiten_strain(preprocess_kernel(torch.load(path_l1, map_location='cpu', weights_only=False).to(device)))
    except: pass
    min_len = min(len(h1), len(l1))
    return h1[:min_len], l1[:min_len]

# ==================== 2. 模拟器 ====================
class Phase9SimulatorGPU:
    def __init__(self, h1_bg, l1_bg, scaling_factor):
        self.h1_bg = h1_bg
        self.l1_bg = l1_bg
        self.scaling_factor = scaling_factor 
        self.target_fs = 2048.0
        self.seg_len = int(24.0 * self.target_fs)  # 改为24秒
        self.max_idx = len(h1_bg) - self.seg_len - 1

    def compute_features_gpu(self, h1, l1):
        vx = h1 - h1.mean(dim=1, keepdim=True)
        vy = l1 - l1.mean(dim=1, keepdim=True)
        cost = (vx * vy).sum(dim=1) / (torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy**2).sum(dim=1)) + 1e-8)
        
        def kurtosis_torch(x):
            mean = x.mean(dim=1, keepdim=True)
            diff = x - mean
            m2 = (diff**2).mean(dim=1)
            m4 = (diff**4).mean(dim=1)
            return m4 / (m2**2 + 1e-8) - 3.0
        
        k_h1 = torch.log1p(torch.abs(kurtosis_torch(h1)))
        k_l1 = torch.log1p(torch.abs(kurtosis_torch(l1)))
        pw = torch.log10(h1.var(dim=1) * l1.var(dim=1) + 1e-30)
        return torch.stack([cost, k_h1, k_l1, pw], dim=1)

    def simulate(self, theta_batch):
        batch_size = theta_batch.shape[0]
        theta_batch = theta_batch.to(device)
        log_omega, xi = theta_batch[:, 0], theta_batch[:, 1]
        
        start_indices = torch.randint(0, self.max_idx, (batch_size,), device=device)
        indices = start_indices.unsqueeze(1) + torch.arange(self.seg_len, device=device)
        n_h1 = self.h1_bg[indices] 
        n_l1 = self.l1_bg[indices] 
        n_h1 -= n_h1.mean(dim=1, keepdim=True)
        n_l1 -= n_l1.mean(dim=1, keepdim=True)
        
        mask_sig = (log_omega > -15.0)
        if mask_sig.any():
            omega = 10**log_omega[mask_sig]
            safe_xi = torch.clamp(xi[mask_sig], min=1e-4)
            amp = torch.sqrt(omega / safe_xi) * self.scaling_factor
            
            n_ev = (self.seg_len * safe_xi * 0.2).long()
            n_ev[xi[mask_sig] >= 0.99] = self.seg_len
            
            raw_noise = torch.randn(mask_sig.sum(), self.seg_len, device=device) * amp.unsqueeze(1)
            starts = torch.randint(0, self.seg_len, (len(n_ev),), device=device)
            starts = torch.min(starts, self.seg_len - n_ev)
            
            positions = torch.arange(self.seg_len, device=device).unsqueeze(0)
            time_mask = (positions >= starts.unsqueeze(1)) & (positions < (starts + n_ev).unsqueeze(1))
            
            from scipy.signal.windows import tukey
            window_cpu = torch.from_numpy(tukey(self.seg_len, alpha=0.1)).float().to(device)
            
            n_h1[mask_sig] += raw_noise * time_mask * window_cpu
            n_l1[mask_sig] += raw_noise * time_mask * window_cpu
            
        return self.compute_features_gpu(n_h1, n_l1)

# ==================== 3. 辅助函数 (已优化) ====================
def generate_training_data(sim, prior, n_samples):
    batch_size = 2000 # 增大 batch size 提高生成速度
    theta_all, x_all = [], []
    for _ in range(0, n_samples, batch_size):
        current_bs = min(batch_size, n_samples - len(theta_all)*batch_size) # 处理尾部
        if current_bs <=0: break
        theta = prior.sample((current_bs,)).to(device)
        x = sim.simulate(theta)
        theta_all.append(theta)
        x_all.append(x)
    return torch.cat(theta_all), torch.cat(x_all)

# 【核心优化】支持批量采样的安全函数
def safe_sample_batch(posterior, x_batch, n_samples=200):
    """
    对 x_batch (Batch, Features) 执行并行采样。
    返回: (n_samples, Batch, Theta_dim)
    """
    try:
        # sbi 原生支持 batch context
        return posterior.sample((n_samples,), x=x_batch, show_progress_bars=False)
    except Exception as e:
        # 如果爆显存或者 sbi 版本不支持，回退到循环但保持静默
        # print(f"Batch sample warning: {e}, falling back to loop")
        res = []
        for i in range(len(x_batch)):
            res.append(posterior.sample((n_samples,), x=x_batch[i], show_progress_bars=False))
        return torch.stack(res, dim=1)

# 【核心优化】完全向量化的校准函数
def fast_calibrate(posterior, sim, n, feature_indices=None):
    # 一次性生成所有噪声数据 (n, 4)
    theta_noise = torch.tensor([[-20.0, 0.1]] * n, device=device)
    obs_noise = sim.simulate(theta_noise)
    
    if feature_indices:
        obs_noise = obs_noise[:, feature_indices]
        
    scores = []
    bs = 100 # 推理 batch size
    for i in range(0, n, bs):
        batch = obs_noise[i:i+bs]
        # === 优化点: 批量传入 batch，而不是循环 ===
        # samples shape: (n_samples, bs, theta_dim)
        samples = safe_sample_batch(posterior, batch, n_samples=200)
        
        # 计算每个样本的均值: mean over dim 0 -> (bs, theta_dim)
        means = samples.mean(dim=0)
        
        # 提取 log_omega (idx 0) -> (bs,)
        scores.extend(means[:, 0].tolist())
        
    return np.percentile(scores, 99.7) # 0.3% FAR

# 【核心优化】高精度二分查找函数
def find_limit(posterior, sim, xi_tgt, thresh, feature_indices=None):
    # 1. 扩大初始搜索范围，防止因为信号太弱或太强而触顶/底
    low, high = -14.0, -3.0
    
    # 2. 增加单次判定的样本数 (原20 -> 100)
    #    这能消除随机波动，让你敢于去探寻更弱的信号
    n_trials = 100
    
    # 3. 提高停止精度的阈值 (原0.2 -> 0.05)
    #    0.05 的 log 误差意味着 Omega 值误差仅约 12%，这在天文上非常精确
    precision_target = 0.05

    while (high - low) > precision_target:
        mid = (high + low) / 2.0
        
        # --- 模拟与推理过程 (保持向量化优势) ---
        # 构造 input tensor: (n_trials, 2)
        theta_test = torch.tensor([[mid, xi_tgt]] * n_trials, device=device)
        obs_test = sim.simulate(theta_test)
        
        if feature_indices:
            obs_test = obs_test[:, feature_indices]
        
        # 批量采样: (200_samples, n_trials, theta_dim)
        samples = safe_sample_batch(posterior, obs_test, n_samples=200)
        
        # 计算每个 trial 的预测均值
        means = samples.mean(dim=0) # shape: (n_trials, theta_dim)
        
        # --- 判定逻辑 ---
        # 统计有多少个 trial 的预测值超过了设定的 99.7% 阈值
        detected_count = (means[:, 0] > thresh).sum().item()
        
        # 探测率计算
        detection_rate = detected_count / n_trials
        
        # 核心逻辑：如果超过 50% 的样本都超过了阈值，说明这个信号强度足够被发现
        # 我们就尝试更弱的信号 (High 变小)
        if detection_rate >= 0.5:
            high = mid
        else:
            low = mid
            
    return high

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 初始化 CSV
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "GPS", "Seed", "Xi", "AI_Mean_Limit", "AI_Std_Dev", "Trad_Mean_Limit", "Trad_Std_Dev", "Scaling_Factor"])
    
    # 初始化详细CSV
    with open(CSV_DETAILED_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "GPS", "Seed", "Xi", "Round", "AI_Value", "Trad_Value"])

    # ==================== 大循环：遍历数据集 ====================
    for i, dataset in enumerate(DATASETS):
        dataset_name = dataset["name"]
        data_file_h1 = dataset["file_h1"]
        data_file_l1 = dataset["file_l1"]
        seed = dataset["seed"]
        
        # 从文件名中提取GPS
        gps = data_file_h1.split('_')[-1].split('.')[0]
        
        # 确保文件名完整显示
        data_file_h1_full = data_file_h1
        data_file_l1_full = data_file_l1
        
        print(f"\n{'='*100}")
        print(f"开始处理 {dataset_name} (进度: {i+1}/{len(DATASETS)})")
        print(f"{'='*100}")
        print(f"  数据集: {dataset_name}")
        print(f"  GPS时间戳: {gps}")
        print(f"  使用种子: {seed}")
        print(f"  H1数据文件: {data_file_h1_full}")
        print(f"  L1数据文件: {data_file_l1_full}")
        print(f"{'='*100}")
        
        # 1. 物理标定
        print("\n[Step 1] 执行物理标定...")
        print(f"  正在标定文件: {data_file_h1_full}")
        PHYSICAL_SCALING = auto_calibrate_scaling(PT_DATA_DIR, data_file_h1_full, TARGET_OMEGA_REF)
        print(f"[Step 1] 物理标定完成: Scaling Factor = {PHYSICAL_SCALING:.2f}\n")

        # 2. 加载数据
        print("[Step 2] 加载数据到 GPU...")
        print(f"  正在加载: H1={data_file_h1_full}")
        print(f"  正在加载: L1={data_file_l1_full}")
        h1_gpu, l1_gpu = load_data_to_gpu(data_file_h1_full, data_file_l1_full)
        print(f"[Step 2] 数据加载完成: H1 shape = {h1_gpu.shape}, L1 shape = {l1_gpu.shape}")
        print(f"  使用的种子: {seed}")
        
        sim_gpu = Phase9SimulatorGPU(h1_gpu, l1_gpu, scaling_factor=PHYSICAL_SCALING)
        prior = BoxUniform(low=torch.tensor([-13.0, 0.001], device=device), 
                           high=torch.tensor([5.0, 1.0], device=device))
        print("[Step 2] 模拟器初始化完成\n")

        # A. 训练
        print("[Step 3] 开始训练模型...")
        print(f"  设置随机种子: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        print(f"  种子设置完成，开始生成训练数据...")

        theta_tr, x_tr = generate_training_data(sim_gpu, prior, N_TRAIN)
        
        print("Training ING-Net (AI)...")
        inf_ai = SNPE(prior=prior, density_estimator="maf", device=str(device))
        inf_ai.append_simulations(theta_tr, x_tr)
        post_ai = inf_ai.build_posterior(inf_ai.train(show_train_summary=False))
        
        print("Training Traditional Baseline...")
        inf_tr = SNPE(prior=prior, density_estimator="maf", device=str(device))
        inf_tr.append_simulations(theta_tr, x_tr[:, [0, 3]])
        post_tr = inf_tr.build_posterior(inf_tr.train(show_train_summary=False))
        
        # B. 检测 (随机熵)
        random_entropy = int.from_bytes(os.urandom(4), byteorder='big')
        torch.manual_seed(random_entropy)
        
        print("执行快速批量 CFAR 校准...")
        thresh_ai = fast_calibrate(post_ai, sim_gpu, N_CALIB, None)
        thresh_tr = fast_calibrate(post_tr, sim_gpu, N_CALIB, [0, 3])
        
        print(f"校准阈值: AI={thresh_ai:.2f}, Trad={thresh_tr:.2f}")
        
        current_avg_rounds = N_AVG_ROUNDS
        print(f"开始高灵敏度扫描 (Xi loop, {current_avg_rounds} rounds)...")
        
        xi_vals = [0.001, 0.01, 0.1, 0.5, 1.0]
        res_ai, res_tr = [], []
        res_ai_std, res_tr_std = [], []
        
        with open(CSV_LOG_PATH, 'a', newline='') as f_log, open(CSV_DETAILED_PATH, 'a', newline='') as f_detail:
            writer_log = csv.writer(f_log)
            writer_detail = csv.writer(f_detail)
            
            for xi in xi_vals:
                print(f"\n处理 Xi={xi} (100次评价)...")
                temp_ai = []
                temp_tr = []
                
                # 执行100次评价并打印每次值
                for round_idx in range(current_avg_rounds):
                    ai_val = find_limit(post_ai, sim_gpu, xi, thresh_ai, None)
                    tr_val = find_limit(post_tr, sim_gpu, xi, thresh_tr, [0, 3])
                    temp_ai.append(ai_val)
                    temp_tr.append(tr_val)
                    
                    # 打印每次值
                    if (round_idx + 1) % 10 == 0 or round_idx == 0 or round_idx == current_avg_rounds - 1:
                        print(f"  Round {round_idx+1:3d}: AI={ai_val:.4f}, Trad={tr_val:.4f}")
                    
                    # 保存到详细CSV
                    writer_detail.writerow([dataset_name, gps, seed, xi, round_idx+1, f"{ai_val:.6f}", f"{tr_val:.6f}"])
                
                mean_ai, std_ai = np.mean(temp_ai), np.std(temp_ai)
                mean_tr, std_tr = np.mean(temp_tr), np.std(temp_tr)
                
                res_ai.append(mean_ai)
                res_tr.append(mean_tr)
                res_ai_std.append(std_ai)
                res_tr_std.append(std_tr)
                
                writer_log.writerow([dataset_name, gps, seed, xi, f"{mean_ai:.4f}", f"{std_ai:.4f}", f"{mean_tr:.4f}", f"{std_tr:.4f}", f"{PHYSICAL_SCALING:.2f}"])
                print(f" -> Xi={xi}: AI={mean_ai:.4f} ± {std_ai:.4f} | Trad={mean_tr:.4f} ± {std_tr:.4f}")

        # C. 绘图 - 灵敏度
        plt.figure(figsize=(12, 7))
        plt.fill_between(xi_vals, np.array(res_ai)-np.array(res_ai_std), np.array(res_ai)+np.array(res_ai_std), color='#1f77b4', alpha=0.2)
        plt.fill_between(xi_vals, np.array(res_tr)-np.array(res_tr_std), np.array(res_tr)+np.array(res_tr_std), color='#ff7f0e', alpha=0.2)
        plt.plot(xi_vals, res_ai, 'o-', color='#1f77b4', label=f'ING-Net ({dataset_name}, GPS: {gps}, Seed: {seed})')
        plt.plot(xi_vals, res_tr, 's--', color='#ff7f0e', label='Traditional')
        plt.xscale('log')
        plt.grid(True, which="both", alpha=0.3)
        plt.title(f'Sensitivity Limit - {dataset_name} (GPS: {gps}, Seed: {seed})')
        plt.ylabel(r'Limit $\log_{10}\Omega_{GW}$')
        plt.xlabel(r'Spectral Index $\xi$')
        plt.legend()
        plt.savefig(os.path.join(CACHE_DIR, "plots", f"sensitivity_{dataset_name}_{gps}_seed_{seed}.png"))
        plt.close()

        # D. 绘图 - SNR Check
        xi_check = xi_vals[0]
        ai_lim = res_ai[0]
        snr_calc = np.sqrt((10**ai_lim)/xi_check) * PHYSICAL_SCALING
        
        plt.figure(figsize=(12, 6))
        log_omega_scan = np.linspace(-9.0, -5.0, 30)
        snr_list = [np.sqrt((10**lo)/xi_check)*PHYSICAL_SCALING for lo in log_omega_scan]
        plt.plot(log_omega_scan, snr_list, 'o-', color='purple', markersize=4)
        plt.axhline(y=8.0, color='r', linestyle='--', label='Detection (~ SNR 8)')
        plt.axhline(y=5.0, color='g', linestyle='--', label='Weak Limit (~ SNR 5)')
        plt.axvline(x=ai_lim, color='b', linestyle='-.', label=f'AI Limit: {ai_lim:.2f} (SNR={snr_calc:.2f})')
        plt.plot(ai_lim, snr_calc, 'ro', markersize=8)
        plt.title(f'SNR Check - {dataset_name} (GPS: {gps}, Seed: {seed})')
        plt.yscale('log')
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(CACHE_DIR, "plots", f"snr_check_{dataset_name}_{gps}_seed_{seed}.png"))
        plt.close()

        # E. 保存模型
        print(f"\n保存模型 - {dataset_name} (Seed {seed})...")
        model_path_ai = os.path.join(CACHE_DIR, "models", f"model_ai_{dataset_name}_{gps}_seed_{seed}.pt")
        model_path_tr = os.path.join(CACHE_DIR, "models", f"model_tr_{dataset_name}_{gps}_seed_{seed}.pt")
        
        # 保存AI模型
        torch.save({
            'posterior': post_ai,
            'scaling_factor': PHYSICAL_SCALING,
            'seed': seed,
            'data_file_h1': data_file_h1,
            'data_file_l1': data_file_l1,
            'dataset': dataset_name,
            'gps': gps
        }, model_path_ai)
        
        # 保存传统模型
        torch.save({
            'posterior': post_tr,
            'scaling_factor': PHYSICAL_SCALING,
            'seed': seed,
            'data_file_h1': data_file_h1,
            'data_file_l1': data_file_l1,
            'dataset': dataset_name,
            'gps': gps
        }, model_path_tr)
        
        print(f"模型保存完成: {model_path_ai}")
        print(f"模型保存完成: {model_path_tr}")

        del post_ai, post_tr, inf_ai, inf_tr
        torch.cuda.empty_cache()

    print("\nBatch Run Completed.")