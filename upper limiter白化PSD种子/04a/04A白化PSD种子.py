
# @title Phase 11 (Final): Multi-Seed Training (Locked) & Random Testing (Unlocked)
import os
import sys
import warnings
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, pearsonr
import scipy.signal as signal
import re
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from tqdm import tqdm
import time

# --- 1. 基础配置 ---
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# [路径配置] O4a 数据
# ==========================================
# GPS: 1374973140
H1_FILE = r"C:\Users\20466\Desktop\upper limiter\ligo_o4a_data\O4a_H1_1374973140.pt"
L1_FILE = r"C:\Users\20466\Desktop\upper limiter\ligo_o4a_data\O4a_L1_1374973140.pt"

# [修改] 结果保存文件夹
RESULTS_DIR = os.path.join(os.getcwd(), "Results_O4a_MultiSeed_Final")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

print(f"[Phase 11 - Multi-Seed O4a Analysis] 开始运行 | 设备: {device}")
print(f"数据路径: {os.path.dirname(H1_FILE)}")
print(f"结果保存: {RESULTS_DIR}")

# --- 科学标定因子计算 ---
def get_scientific_factor(fs=2048, f_ref=25.0):
    H0_SI = 2.1927e-18
    PI = np.pi
    psd_val = (3 * H0_SI**2) / (10 * PI**2 * f_ref**3 )
    variance = psd_val * (fs / 2 )
    return np.sqrt(variance)

PHYSICAL_FACTOR_OMEGA_1 = get_scientific_factor(fs=2048, f_ref=25.0)
print(f"科学标定因子: {PHYSICAL_FACTOR_OMEGA_1:.3e}")

# ==========================================
# [工具函数] 种子管理 (关键部分)
# ==========================================
def set_seed(seed):
    """
    [训练阶段] 严格锁定随机种子。
    确保每次使用该种子训练时，数据生成和神经网络权重初始化完全一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 保证卷积算法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n[System] Random Seed LOCKED to: {seed} (Training Phase)")

def release_seed():
    """
    [测试阶段] 解除种子锁定。
    恢复基于系统时钟的随机性，确保测试集切片选取是公正随机的。
    """
    # 主要是 numpy 用于切片索引，将其设为 None 即可恢复随机
    np.random.seed(None) 
    random.seed()
    print(f"\n[System] Random Seed RELEASED (Testing Phase - Fully Random)")

# ==========================================
# [核心] 鲁棒白化函数
# ==========================================
def robust_whiten(data, fs=2048, fftlength=2.0):
    """
    结合了 'robust_sigma_psd' (抗Glitch) 和 'Whitening' (频谱扁平化)。
    """
    # 1. 计算鲁棒 PSD (Median Welch)
    nperseg = int(fftlength * fs)
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, average='median')
    asd = np.sqrt(psd)
    
    # 2. 频域白化
    data_fft = np.fft.rfft(data)
    fft_freqs = np.fft.rfftfreq(len(data), d=1.0/fs)
    
    # 对数域插值
    interp_asd = np.exp(np.interp(np.log(fft_freqs[1:]), np.log(freqs[1:]), np.log(asd[1:])))
    interp_asd = np.insert(interp_asd, 0, interp_asd[0])
    
    # 白化
    whitened_fft = data_fft / (interp_asd + 1e-30)
    whitened_data = np.fft.irfft(whitened_fft, n=len(data))
    
    # 3. 最终归一化
    return whitened_data / np.std(whitened_data)

# ==========================================
# 2. 数据准备
# ==========================================
print("\n>>> [1/7] 加载 O4a 数据文件...")

if not os.path.exists(H1_FILE) or not os.path.exists(L1_FILE):
    print("错误：未找到 O4a 数据文件！")
    sys.exit()

try:
    first_gps = int(re.search(r'\d{10}', os.path.basename(H1_FILE)).group())
except:
    first_gps = 1374973140

full_h1 = torch.load(H1_FILE, map_location='cpu')
full_l1 = torch.load(L1_FILE, map_location='cpu')

if isinstance(full_h1, torch.Tensor): full_h1 = full_h1.numpy().flatten()
if isinstance(full_l1, torch.Tensor): full_l1 = full_l1.numpy().flatten()

min_len = min(len(full_h1), len(full_l1))
mid_point = min_len // 2
train_h1, train_l1 = full_h1[:mid_point], full_l1[:mid_point]
test_h1, test_l1 = full_h1[mid_point:], full_l1[mid_point:]
print(f"    数据就绪: 训练集 {len(train_h1)/2048:.1f}s | 测试集 {len(test_h1)/2048:.1f}s")

# ==========================================
# 3. 模拟器
# ==========================================
def feature_simulator(theta_batch):
    # 注意：此时的 np.random 状态由外部的 set_seed 控制
    if isinstance(theta_batch, np.ndarray): theta_batch = torch.from_numpy(theta_batch)
    batch_stats = []
    seg_len = 8192
    fs = 2048
    shift_samples = 1 * fs
    max_idx = len(train_h1) - seg_len
    
    global PHYSICAL_FACTOR_OMEGA_1
    
    for theta in theta_batch:
        log10_omega = theta[0].item()
        xi = theta[1].item()
        
        # 1. 采样背景 (受 Seed 控制)
        idx_h1 = np.random.randint(0, max_idx)
        idx_l1 = np.random.randint(0, max_idx)
        while abs(idx_l1 - idx_h1) < shift_samples:
             idx_l1 = np.random.randint(0, max_idx)
        
        bg_h1 = train_h1[idx_h1 : idx_h1 + seg_len].copy()
        bg_l1 = train_l1[idx_l1 : idx_l1 + seg_len].copy()
        
        # 2. 计算物理信号
        omega = 10**log10_omega
        safe_xi = np.max([xi, 1e-4])
        amp_phys = np.sqrt(omega / safe_xi) * PHYSICAL_FACTOR_OMEGA_1
        
        n_events = int(seg_len * safe_xi * 0.2)
        sig_pattern = np.zeros(seg_len)
        if n_events > 0:
            idx = np.random.randint(0, seg_len, n_events)
            sig_pattern[idx] += np.random.normal(0, 1.0, n_events)
            
        sig_phys = sig_pattern * amp_phys
        
        # 3. 注入
        raw_h1 = bg_h1 + sig_phys
        raw_l1 = bg_l1 + sig_phys
        
        # 4. 鲁棒白化
        d_h1 = robust_whiten(raw_h1, fs=fs)
        d_l1 = robust_whiten(raw_l1, fs=fs)
        
        # 5. 特征提取
        cc, _ = pearsonr(d_h1, d_l1)
        k_h1 = np.log1p(np.abs(kurtosis(d_h1)))
        k_l1 = np.log1p(np.abs(kurtosis(d_l1)))
        p = np.log10(np.var(d_h1) * np.var(d_l1) + 1e-30)
        
        batch_stats.append(torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32))
        
    return torch.stack(batch_stats)

# ==========================================
# 4 & 5. 核心逻辑：3种子训练 + 10轮随机测试
# ==========================================
# 定义 3 个随机种子
SEEDS = [42, 1024, 2024]

prior = BoxUniform(low=torch.tensor([-25.0, 0.001], device=device), 
                   high=torch.tensor([-5.0, 1.0], device=device))

print(f"\n>>> [2/7] 开始全流程 (Seeds: {SEEDS}) ...")

for i, seed in enumerate(SEEDS):
    print(f"\n{'#'*30}")
    print(f"###  Processing Seed Group {seed} ({i+1}/3)  ###")
    print(f"{'#'*30}")
    
    # ----------------------------------------------------
    # Part A: 训练阶段 (Training) - 必须锁定种子
    # ----------------------------------------------------
    set_seed(seed)
    
    print(f">>> [Seed {seed}] Generating Simulation Data (15,000 sims)...")
    theta_train, x_train = simulate_for_sbi(feature_simulator, proposal=prior, num_simulations=15000, num_workers=1)
    
    # 训练 AI 模型
    print(f">>> [Seed {seed}] Training ING-Net...")
    inference_ai = SNPE(prior=prior, density_estimator="maf", device=device)
    inference_ai.append_simulations(theta_train, x_train)
    de_ai = inference_ai.train(show_train_summary=False)
    post_ai = inference_ai.build_posterior(de_ai, sample_with='direct')

    # 训练 Traditional 模型
    print(f">>> [Seed {seed}] Training Traditional Baseline...")
    x_train_trad = x_train[:, [0, 3]] 
    inference_trad = SNPE(prior=prior, density_estimator="maf", device=device)
    inference_trad.append_simulations(theta_train, x_train_trad)
    de_trad = inference_trad.train(show_train_summary=False)
    post_trad = inference_trad.build_posterior(de_trad, sample_with='direct')

    # 保存模型 (带 Seed 后缀)
    timestamp = time.strftime('%Y%m%d')
    model_dir = os.path.join(RESULTS_DIR, f'Models_Seed_{seed}_{timestamp}')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    torch.save(de_ai, os.path.join(model_dir, f'model_ai_seed_{seed}.pth'))
    torch.save(de_trad, os.path.join(model_dir, f'model_trad_seed_{seed}.pth'))
    print(f"    ✅ 模型已保存至: {model_dir}")
    
    # ----------------------------------------------------
    # Part B: 测试阶段 (Testing) - 必须释放种子
    # ----------------------------------------------------
    release_seed() # <--- 关键：确保测试是随机的
    
    print(f">>> [Seed {seed}] 开始 10 轮随机性测试 (无固定种子)...")
    
    NUM_ROUNDS = 10
    TESTS_PER_ROUND = 50
    seg_len = 8192
    max_idx_test = len(test_h1) - seg_len

    # 对该模型进行 10 轮测试
    for round_idx in range(1, NUM_ROUNDS + 1):
        ul_ai_list = []
        ul_trad_list = []
        
        for i in tqdm(range(TESTS_PER_ROUND), desc=f"Testing (Round {round_idx})", leave=False):
            # 随机抽取切片 (因为 release_seed 了，所以这里是真正的随机)
            start_idx = np.random.randint(0, max_idx_test)
            slice_h1 = test_h1[start_idx : start_idx + seg_len].copy()
            slice_l1 = test_l1[start_idx : start_idx + seg_len].copy()
            
            # 鲁棒白化
            slice_h1 = robust_whiten(slice_h1, fs=2048)
            slice_l1 = robust_whiten(slice_l1, fs=2048)
            
            # 特征提取
            cc, _ = pearsonr(slice_h1, slice_l1)
            k_h1 = np.log1p(np.abs(kurtosis(slice_h1)))
            k_l1 = np.log1p(np.abs(kurtosis(slice_l1)))
            p = np.log10(np.var(slice_h1) * np.var(slice_l1) + 1e-30)
            
            obs_full = torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32).to(device)
            obs_trad = torch.tensor([cc, p], dtype=torch.float32).to(device)
            
            # 推断
            s_ai = post_ai.sample((1000,), x=obs_full, show_progress_bars=False)
            ul_ai_val = np.percentile(s_ai.cpu().numpy()[:, 0], 95)
            ul_ai_list.append(ul_ai_val)
            
            s_trad = post_trad.sample((1000,), x=obs_trad, show_progress_bars=False)
            ul_trad_val = np.percentile(s_trad.cpu().numpy()[:, 0], 95)
            ul_trad_list.append(ul_trad_val)

        # 计算结果
        mean_ai = np.mean(ul_ai_list)
        mean_trad = np.mean(ul_trad_list)
        improvement = 10**mean_trad / 10**mean_ai
        
        print(f"    > Seed {seed} | Round {round_idx}: Trad=10^{mean_trad:.2f} | AI=10^{mean_ai:.2f} | Improvement={improvement:.2f}x")

        # 绘图
        plt.figure(figsize=(8, 6))
        plt.hist(ul_ai_list, bins=15, density=True, alpha=0.5, color='blue', label=f'ING-Net (Model Seed {seed})')
        plt.hist(ul_trad_list, bins=15, density=True, alpha=0.5, color='orange', label='Traditional')
        plt.axvline(mean_ai, color='blue', linestyle='--', linewidth=2)
        plt.axvline(mean_trad, color='orange', linestyle='--', linewidth=2)
        plt.xlabel(r'95% Upper Limit ($\log_{10}\Omega$)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Stability Test - Model Seed {seed} - Round {round_idx}/10\n(O4a Data - Random Test Slices)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        # 命名格式: Seed_种子号_Round_轮次_gps.png
        save_path = os.path.join(RESULTS_DIR, f"Seed_{seed}_Round_{round_idx:02d}_gps_{first_gps}.png")
        plt.savefig(save_path)
        plt.close() # 必须关闭，防止生成30张图导致内存溢出

print("\n>>> =========================================")
print(">>> 全部任务完成！Summary:")
print(">>> 1. 训练了 3 个不同种子的模型 (42, 1024, 2024) 并已保存。")
print(">>> 2. 对每个模型进行了 10 轮 *完全随机* 的测试。")
print(">>> 3. 生成了 30 张分析图。")
print(f">>> 结果路径: {RESULTS_DIR}")
print(">>> =========================================")

