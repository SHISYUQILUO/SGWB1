# @title Phase 9 (O4a): Batch Training & Detection (15 Seeds + CSV + SNR Calculation + Weak Limit Line)
# @markdown **Batch Mode Features:**
# @markdown 1. **15 Training Seeds:** Iterates through 15 distinct random seeds.
# @markdown 2. **Explicit SNR Calculation:** Calculates and displays the exact SNR at the detection limit.
# @markdown 3. **Full Reference Lines:** Includes both Red (SNR~8) and Green (SNR~5) reference lines.
# @markdown 4. **Full Logging:** Saves models, plots, and CSV reports.

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

print("=== P904a_Batch_15Seeds_CalcSNR_Fixed.py 启动 ===")

# ==================== 配置区域 ====================
# 输入数据路径
PT_DATA_DIR = "."  # 使用当前目录
# 输出路径 (O4a 1369205931)
CACHE_DIR = "./output_batch_run_o4a_1369205931"
# 自动创建输出目录
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "plots"), exist_ok=True)

# 指定具体文件
DATA_FILE_H1 = "O4a_H1_1369205931.pt"
DATA_FILE_L1 = "O4a_L1_1369205931.pt"

N_TRAIN = 20000            
N_CALIB = 1000            
TARGET_OMEGA_REF = 1e-7   
N_AVG_ROUNDS = 20         

# 只保留种子555
TRAIN_SEEDS = [555]

# CSV 日志文件路径
CSV_LOG_PATH = os.path.join(CACHE_DIR, "batch_results_log.csv")

# 检查 GPU
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
    print(f"[DEBUG] 正在尝试加载文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        from gwpy.timeseries import TimeSeries
    except ImportError:
        print("[DEBUG] ❌ 错误: 未安装 gwpy 库。")
        raise

    H0_SI = 2.1927e-18
    TARGET_OMEGA = 1e-7
    FS = 2048.0  
    
    try:
        raw_data = torch.load(file_path, map_location='cpu', weights_only=False)
        if isinstance(raw_data, torch.Tensor):
            raw_data = raw_data.numpy()
        raw_data = raw_data.astype(np.float64)
    except Exception as e:
        print(f"[DEBUG] ❌ 数据加载失败: {e}")
        raise

    try:
        ts = TimeSeries(raw_data, sample_rate=4096.0)
        ts = ts.resample(FS)
        ts = ts.highpass(15.0)
        white_ts = ts.whiten()
        if np.isnan(white_ts.value).any():
             raise ValueError("Whitening produced NaNs")
    except Exception as e:
        print(f"[DEBUG] ❌ 信号处理失败: {e}")
        raise

    try:
        safe_data = white_ts[int(4*FS):-int(4*FS)]
        freqs, Pxx = scipy.signal.welch(safe_data.value, fs=FS, nperseg=int(4*FS), average='median')
        mask = (freqs >= 20) & (freqs <= 1000)
        f_int = freqs[mask]
        S_gw_physical = (3 * H0_SI**2 / (10 * np.pi**2)) * (TARGET_OMEGA / f_int**3)
        _, Pxx_raw = scipy.signal.welch(ts.value, fs=FS, nperseg=int(4*FS), average='median')
        psd_raw = Pxx_raw[mask]
        integrand = S_gw_physical / (psd_raw + 1e-50)
        df = f_int[1] - f_int[0]
        variance_ratio = np.sum(integrand) * df * 2
        scaling_factor = np.sqrt(variance_ratio) / np.sqrt(TARGET_OMEGA)
        print(f"[DEBUG] ✅ 标定成功! Scaling Factor: {scaling_factor}")
        return scaling_factor
    except Exception as e:
        print(f"[DEBUG] ❌ 物理计算失败: {e}")
        raise

def auto_calibrate_scaling(data_dir, filename, target_omega=1e-7):
    print("-" * 40)
    print("开始自动标定流程...")
    h1_file_path = os.path.join(data_dir, filename)
    try:
        calculated_scaling = get_scaling_factor_aligned_with_training(h1_file_path)
        if np.isinf(calculated_scaling) or np.isnan(calculated_scaling):
            return 2500.0
        return calculated_scaling
    except Exception as e:
        traceback.print_exc()
        return 2500.0

# ==================== 核心逻辑：白化与ASD计算 ====================
def compute_asd(data, fs=2048.0, nperseg=None):
    if nperseg is None: nperseg = int(4 * fs)
    f, Pxx = scipy.signal.welch(data, fs=fs, nperseg=nperseg, average='median')
    return f, np.sqrt(Pxx)

def whiten_strain(strain_tensor, fs=2048.0):
    data_cpu = strain_tensor.cpu().numpy()
    f_welch, asd = compute_asd(data_cpu, fs=fs)
    n_samples = len(data_cpu)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    data_fft = np.fft.rfft(data_cpu)
    asd_interp = np.interp(freqs, f_welch, asd)
    asd_interp[asd_interp < 1e-30] = 1e-30
    data_fft_white = data_fft / asd_interp
    data_white = np.fft.irfft(data_fft_white, n=n_samples)
    return torch.from_numpy(data_white).float().to(device)

# ==================== 1. 数据加载 ====================
def load_data_to_gpu(h1_name, l1_name):
    expected_length = int(4096 * 2048.0)
    filenames = [h1_name, l1_name]
    loaded = {}
    for fname in filenames:
        if "H1" in fname: det = "H1"
        elif "L1" in fname: det = "L1"
        else: continue
        path = os.path.join(PT_DATA_DIR, fname)
        if os.path.exists(path):
            try:
                data = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(data, np.ndarray): data = torch.from_numpy(data)
                processed_data = preprocess_kernel(data, input_fs=4096.0, target_fs=2048.0)
                raw_tensor = processed_data.to(device)
                loaded[det] = whiten_strain(raw_tensor)
            except: continue
    h1 = loaded.get('H1', torch.randn(expected_length, device=device))
    l1 = loaded.get('L1', torch.randn(expected_length, device=device))
    min_len = min(len(h1), len(l1))
    return h1[:min_len], l1[:min_len]

# ==================== 2. 模拟器 ====================
class Phase9SimulatorGPU:
    def __init__(self, h1_bg, l1_bg, scaling_factor):
        self.h1_bg = h1_bg
        self.l1_bg = l1_bg
        self.scaling_factor = scaling_factor 
        self.target_fs = 2048.0
        self.seg_len = int(4.0 * self.target_fs)
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
        
        n_h1 = n_h1 - n_h1.mean(dim=1, keepdim=True)
        n_l1 = n_l1 - n_l1.mean(dim=1, keepdim=True)
        
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

# ==================== 3. 辅助函数 ====================
def generate_training_data(sim, prior, n_samples):
    batch_size = 1000
    theta_all, x_all = [], []
    for _ in range(0, n_samples, batch_size):
        theta = prior.sample((batch_size,)).to(device)
        x = sim.simulate(theta)
        theta_all.append(theta)
        x_all.append(x)
    return torch.cat(theta_all), torch.cat(x_all)

def safe_sample(posterior, x, n_samples=200):
    try:
        return posterior.sample((n_samples,), x=x, show_progress_bars=False)
    except:
        return torch.tensor([[10.0, 0.5]] * n_samples, device=device)

def fast_calibrate(posterior, sim, n, feature_indices=None):
    theta_noise = torch.tensor([[-20.0, 0.1]] * n, device=device)
    obs_noise = sim.simulate(theta_noise)
    scores = []
    bs = 100
    for i in range(0, n, bs):
        batch = obs_noise[i:i+bs]
        if feature_indices: batch = batch[:, feature_indices]
        for j in range(len(batch)):
            s = safe_sample(posterior, batch[j])
            scores.append(s[:, 0].mean().item())
    return np.percentile(scores, 90)

def find_limit(posterior, sim, xi_tgt, thresh, feature_indices=None):
    low, high = -13.0, -1.0
    n_trials = 20
    while (high - low) > 0.2:
        mid = (high + low) / 2.0
        theta_test = torch.tensor([[mid, xi_tgt]] * n_trials, device=device)
        obs_test = sim.simulate(theta_test)
        if feature_indices: obs_test = obs_test[:, feature_indices]
        
        detected = 0
        for i in range(n_trials):
            s = safe_sample(posterior, obs_test[i])
            if s[:, 0].mean() > thresh: detected += 1
        
        if detected >= (n_trials / 2): high = mid
        else: low = mid
    return high

# ==================== 主流程 ====================
if __name__ == "__main__":
    
    # 0. 初始化 CSV (写入表头)
    print(f"初始化 CSV 日志: {CSV_LOG_PATH}")
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Xi", "AI_Mean_Limit", "AI_Std_Dev", "Trad_Mean_Limit", "Trad_Std_Dev", "Scaling_Factor"])

    # 1. 物理标定
    print("\n[Step 1] 执行物理标定...")
    PHYSICAL_SCALING = auto_calibrate_scaling(PT_DATA_DIR, DATA_FILE_H1, TARGET_OMEGA_REF)
    print(f"物理 Scaling Factor: {PHYSICAL_SCALING:.2f}\n")

    # 2. 加载数据
    h1_gpu, l1_gpu = load_data_to_gpu(DATA_FILE_H1, DATA_FILE_L1)
    
    # 初始化模拟器 (数据已在 GPU)
    sim_gpu = Phase9SimulatorGPU(h1_gpu, l1_gpu, scaling_factor=PHYSICAL_SCALING)
    prior = BoxUniform(low=torch.tensor([-13.0, 0.001], device=device), 
                       high=torch.tensor([5.0, 1.0], device=device))

    # ==================== 大循环：遍历 15 个种子 ====================
    for i, seed in enumerate(TRAIN_SEEDS):
        print(f"\n{'='*60}")
        print(f"开始处理 Seed {seed} (进度: {i+1}/{len(TRAIN_SEEDS)})")
        print(f"{'='*60}")
        
        # A. 训练阶段 - 【设置固定种子】
        print(f"设置训练种子: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 生成训练数据
        theta_tr, x_tr = generate_training_data(sim_gpu, prior, N_TRAIN)
        
        # 训练 AI
        print("Training ING-Net (AI)...")
        inf_ai = SNPE(prior=prior, density_estimator="maf", device=str(device))
        inf_ai.append_simulations(theta_tr, x_tr)
        post_ai = inf_ai.build_posterior(inf_ai.train(show_train_summary=False))
        
        # 训练 Traditional (基准)
        print("Training Traditional Baseline...")
        inf_tr = SNPE(prior=prior, density_estimator="maf", device=str(device))
        inf_tr.append_simulations(theta_tr, x_tr[:, [0, 3]])
        post_tr = inf_tr.build_posterior(inf_tr.train(show_train_summary=False))
        
        # 保存模型
        torch.save(post_ai, os.path.join(CACHE_DIR, "models", f"ing_net_seed_{seed}.pt"))
        torch.save(post_tr, os.path.join(CACHE_DIR, "models", f"trad_model_seed_{seed}.pt"))
        
        # B. 检测阶段 - 【强制打乱种子】
        random_entropy = int.from_bytes(os.urandom(4), byteorder='big')
        print(f"⚠️ 检测阶段: 移除种子限制，使用随机熵 {random_entropy} 重置 RNG...")
        
        torch.manual_seed(random_entropy)
        np.random.seed(random_entropy % (2**32))
        random.seed(random_entropy)
        
        print("执行 CFAR 校准 (Calibration)...")
        thresh_ai = fast_calibrate(post_ai, sim_gpu, N_CALIB, None)
        thresh_tr = fast_calibrate(post_tr, sim_gpu, N_CALIB, [0, 3])
        
        # 使用配置的检测次数
        current_avg_rounds = N_AVG_ROUNDS
        
        print(f"开始灵敏度扫描 (Xi loop, Avg {current_avg_rounds} rounds)...")
        xi_vals = [0.001, 0.01, 0.1, 0.5, 1.0]
        res_ai, res_tr = [], []
        res_ai_std, res_tr_std = [], []
        
        # 打开 CSV 追加模式
        with open(CSV_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for xi in xi_vals:
                # AI Loop
                temp_ai = []
                for _ in range(current_avg_rounds):
                    val = find_limit(post_ai, sim_gpu, xi, thresh_ai, None)
                    temp_ai.append(val)
                mean_ai = np.mean(temp_ai)
                std_ai = np.std(temp_ai)
                
                # Trad Loop
                temp_tr = []
                for _ in range(current_avg_rounds):
                    val = find_limit(post_tr, sim_gpu, xi, thresh_tr, [0, 3])
                    temp_tr.append(val)
                mean_tr = np.mean(temp_tr)
                std_tr = np.std(temp_tr)
                
                res_ai.append(mean_ai)
                res_tr.append(mean_tr)
                res_ai_std.append(std_ai)
                res_tr_std.append(std_tr)
                
                writer.writerow([seed, xi, f"{mean_ai:.4f}", f"{std_ai:.4f}", f"{mean_tr:.4f}", f"{std_tr:.4f}", f"{PHYSICAL_SCALING:.2f}"])
                print(f"  -> Xi={xi}: AI={mean_ai:.2f}(±{std_ai:.2f}) | Trad={mean_tr:.2f}")

        # C. 绘图与保存 - 灵敏度曲线
        plt.figure(figsize=(10, 7))
        plt.fill_between(xi_vals, np.array(res_ai) - np.array(res_ai_std), np.array(res_ai) + np.array(res_ai_std), color='#1f77b4', alpha=0.2)
        plt.fill_between(xi_vals, np.array(res_tr) - np.array(res_tr_std), np.array(res_tr) + np.array(res_tr_std), color='#ff7f0e', alpha=0.2)
        plt.plot(xi_vals, res_ai, 'o-', color='#1f77b4', label=f'ING-Net (Seed {seed})')
        plt.plot(xi_vals, res_tr, 's--', color='#ff7f0e', label='Traditional')
        plt.xscale('log')
        plt.grid(True, which="both", alpha=0.3)
        plt.title(f'Sensitivity Limit - Training Seed {seed}')
        plt.ylabel(r'Limit $\log_{10}\Omega_{GW}$')
        plt.xlabel(r'Spectral Index $\xi$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CACHE_DIR, "plots", f"sensitivity_seed_{seed}.png"))
        plt.close()

        # D. 绘图与保存 - SNR 物理校验 (含 SNR 计算与所有参考线)
        # -------------------------------------------------------------------------
        print(f"[Seed {seed}] 生成 SNR 物理一致性检查图...")
        
        # 使用 Xi = 0.001 进行检查
        xi_check_target = xi_vals[0]
        ai_check_limit = res_ai[0] 

        # === 核心修改：计算具体的 SNR ===
        omega_at_limit = 10**ai_check_limit
        calculated_snr = np.sqrt(omega_at_limit / xi_check_target) * PHYSICAL_SCALING
        print(f"  -> 在极限 {ai_check_limit:.2f} 处，计算得到的 SNR = {calculated_snr:.2f}")

        # 模拟 SNR 扫描曲线
        log_omega_scan = np.linspace(-9.0, -5.0, 30)
        snr_scan_list = []
        for lo in log_omega_scan:
             omega = 10**lo
             val = np.sqrt(omega / xi_check_target) * PHYSICAL_SCALING
             snr_scan_list.append(val)

        plt.figure(figsize=(10, 6))
        plt.plot(log_omega_scan, snr_scan_list, 'o-', linewidth=2, color='purple', markersize=4)
        
        # === 修复：添加回 SNR=8 (Red) 和 SNR=5 (Green) 完整的参考线 ===
        plt.axhline(y=8.0, color='r', linestyle='--', label='Theoretical Detection (~ SNR 8)')
        plt.axhline(y=5.0, color='g', linestyle='--', label='Weak Limit (~ SNR 5)')
        
        # 绘制检测线 + 显示 SNR 数值
        plt.axvline(x=ai_check_limit, color='b', linestyle='-.', linewidth=2, 
                    label=f'AI Limit: {ai_check_limit:.2f} (Calc SNR={calculated_snr:.2f})')
        
        # 在交叉点画一个红点，视觉增强
        plt.plot(ai_check_limit, calculated_snr, 'ro', markersize=8, zorder=5)
        
        plt.xlabel(r'Signal Strength $\log_{10}\Omega$')
        plt.ylabel(f'Single Burst SNR (xi={xi_check_target})')
        # 标题也加上 SNR
        plt.title(f'SNR Check - Seed {seed} (Limit SNR = {calculated_snr:.2f})')
        plt.yscale('log')
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        snr_save_path = os.path.join(CACHE_DIR, "plots", f"snr_check_seed_{seed}.png")
        plt.savefig(snr_save_path, dpi=200)
        plt.close()
        print(f"  -> SNR 图已保存: {snr_save_path}")

        # E. 显存清理
        del post_ai
        del post_tr
        del inf_ai
        del inf_tr
        del theta_tr
        del x_tr
        torch.cuda.empty_cache()
        print(f"Seed {seed} 完成。显存已清理。")

    print("\n" + "="*60)
    print("所有种子任务执行完毕！")
    print(f"结果目录: {CACHE_DIR}")
    print("="*60)