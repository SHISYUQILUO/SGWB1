# @title Phase 11 (Final): O3b Batch Training (5 GPS x 5 Seeds)
import os
import sys
import warnings
import numpy as np
import torch
import random
import time
import re
from scipy.stats import kurtosis, pearsonr
import scipy.signal as signal

# [关键设置] 使用非交互式后端，防止画图时弹出窗口卡住程序
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from tqdm import tqdm

# --- 1. 基础配置 ---
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# [配置区]
# ==========================================
# 1. 数据文件夹路径
BASE_DATA_PATH = r"C:\Users\20466\Desktop\upper limiter\ligo_o3b_data"

# 2. 10个 O3b GPS 时间戳 (实际数据文件中的时间戳)
GPS_LIST = [1256775418, 1257223526, 1259365388, 1259640215, 1262044959, 1262159898, 1264822450, 1265080061, 1267354720, 1267465160]

# 3. 10个 训练用随机种子 (用于生成100个不同的模型)
SEEDS_LIST = [42, 101, 2024, 8888, 5678, 31415, 27182, 14142, 73737, 98765]

# 4. 结果保存总目录
RESULTS_DIR = os.path.join(os.getcwd(), "Results_O3b_100Models_Batch")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

print(f"==========================================================")
print(f"   LIGO O3b 批量训练任务启动")
print(f"   目标: 10 GPS x 10 Seeds = 100 Models")
print(f"   设备: {device}")
print(f"   结果目录: {RESULTS_DIR}")
print(f"==========================================================\n")

# --- 科学标定因子 ---
def get_scientific_factor(fs=2048, f_ref=25.0):
    H0_SI = 2.1927e-18
    PI = np.pi
    psd_val = (3 * H0_SI**2) / (10 * PI**2 * f_ref**3 )
    variance = psd_val * (fs / 2 )
    return np.sqrt(variance)

PHYSICAL_FACTOR_OMEGA_1 = get_scientific_factor(fs=2048, f_ref=25.0)

# ==========================================
# [种子控制模块] - 核心逻辑
# ==========================================
def set_seed(seed):
    """
    [训练阶段] 锁定种子。
    目的：确保每次训练产生的模型是确定性的，可复现的。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"   [System] Seed LOCKED to {seed} (Training)")

def release_seed():
    """
    [测试阶段] 释放种子。
    目的：确保测试集切片是完全随机选取的，证明没有“挑数据”。
    """
    np.random.seed(None) 
    # print(f"   [System] Seed RELEASED (Testing - Random Slices)")

# ==========================================
# [鲁棒白化模块]
# ==========================================
def robust_whiten(data, fs=2048, fftlength=2.0):
    nperseg = int(fftlength * fs)
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, average='median')
    asd = np.sqrt(psd)
    
    data_fft = np.fft.rfft(data)
    fft_freqs = np.fft.rfftfreq(len(data), d=1.0/fs)
    
    interp_asd = np.exp(np.interp(np.log(fft_freqs[1:]), np.log(freqs[1:]), np.log(asd[1:])))
    interp_asd = np.insert(interp_asd, 0, interp_asd[0])
    
    whitened_fft = data_fft / (interp_asd + 1e-30)
    whitened_data = np.fft.irfft(whitened_fft, n=len(data))
    
    return whitened_data / np.std(whitened_data)

# ==========================================
# [主循环] 遍历 GPS
# ==========================================
for gps_idx, gps in enumerate(GPS_LIST):
    print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f" PROCESSING GPS TIMESTAMP [{gps_idx+1}/10]: {gps}")
    print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    # 1. 加载数据
    H1_PATH = os.path.join(BASE_DATA_PATH, f"O3b_H1_{gps}.pt")
    L1_PATH = os.path.join(BASE_DATA_PATH, f"O3b_L1_{gps}.pt")
    
    if not os.path.exists(H1_PATH):
        print(f"错误: 文件不存在 {H1_PATH}, 跳过此GPS。")
        continue

    full_h1 = torch.load(H1_PATH, map_location='cpu')
    full_l1 = torch.load(L1_PATH, map_location='cpu')

    if isinstance(full_h1, torch.Tensor): full_h1 = full_h1.numpy().flatten()
    if isinstance(full_l1, torch.Tensor): full_l1 = full_l1.numpy().flatten()

    # 切分训练/测试集
    min_len = min(len(full_h1), len(full_l1))
    mid_point = min_len // 2
    train_h1, train_l1 = full_h1[:mid_point], full_l1[:mid_point]
    test_h1, test_l1 = full_h1[mid_point:], full_l1[mid_point:]
    
    print(f"数据加载完毕. 训练集长度: {len(train_h1)}")

    # ==========================================
    # [内层循环] 遍历种子 (Training Phase)
    # ==========================================
    for seed_idx, seed in enumerate(SEEDS_LIST):
        print(f"\n   ------ Training Seed [{seed_idx+1}/10]: {seed} (GPS {gps}) ------")
        
        # -----------------------------------
        # A. 训练阶段 (LOCKED SEED)
        # -----------------------------------
        set_seed(seed)  # <--- 关键：锁定种子
        
        # 定义模拟器 (需包含在循环内以使用当前的 train_h1)
        def feature_simulator(theta_batch):
            if isinstance(theta_batch, np.ndarray): theta_batch = torch.from_numpy(theta_batch)
            batch_stats = []
            seg_len = 8192
            fs = 2048
            shift_samples = 1 * fs
            max_idx = len(train_h1) - seg_len
            
            for theta in theta_batch:
                log10_omega = theta[0].item()
                xi = theta[1].item()
                
                # 采样背景 (受 Seed 控制，确定性)
                idx_h1 = np.random.randint(0, max_idx)
                idx_l1 = np.random.randint(0, max_idx)
                while abs(idx_l1 - idx_h1) < shift_samples:
                     idx_l1 = np.random.randint(0, max_idx)
                
                bg_h1 = train_h1[idx_h1 : idx_h1 + seg_len].copy()
                bg_l1 = train_l1[idx_l1 : idx_l1 + seg_len].copy()
                
                # 信号生成与注入
                omega = 10**log10_omega
                safe_xi = np.max([xi, 1e-4])
                amp_phys = np.sqrt(omega / safe_xi) * PHYSICAL_FACTOR_OMEGA_1
                
                n_events = int(seg_len * safe_xi * 0.2)
                sig_pattern = np.zeros(seg_len)
                if n_events > 0:
                    idx = np.random.randint(0, seg_len, n_events)
                    sig_pattern[idx] += np.random.normal(0, 1.0, n_events)
                    
                sig_phys = sig_pattern * amp_phys
                
                # 注入并白化
                raw_h1 = bg_h1 + sig_phys
                raw_l1 = bg_l1 + sig_phys
                d_h1 = robust_whiten(raw_h1, fs=fs)
                d_l1 = robust_whiten(raw_l1, fs=fs)
                
                # 特征提取
                cc, _ = pearsonr(d_h1, d_l1)
                k_h1 = np.log1p(np.abs(kurtosis(d_h1)))
                k_l1 = np.log1p(np.abs(kurtosis(d_l1)))
                p = np.log10(np.var(d_h1) * np.var(d_l1) + 1e-30)
                
                batch_stats.append(torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32))
            return torch.stack(batch_stats)

        # 训练过程
        prior = BoxUniform(low=torch.tensor([-25.0, 0.001], device=device), 
                           high=torch.tensor([-5.0, 1.0], device=device))
        
        # 生成数据 (确定性)
        theta_train, x_train = simulate_for_sbi(feature_simulator, proposal=prior, num_simulations=15000, num_workers=-1)
        
        # 训练 AI
        inference_ai = SNPE(prior=prior, density_estimator="maf", device=device)
        inference_ai.append_simulations(theta_train, x_train)
        de_ai = inference_ai.train(show_train_summary=False)
        post_ai = inference_ai.build_posterior(de_ai, sample_with='direct')
        
        # 训练 Traditional
        x_train_trad = x_train[:, [0, 3]] 
        inference_trad = SNPE(prior=prior, density_estimator="maf", device=device)
        inference_trad.append_simulations(theta_train, x_train_trad)
        de_trad = inference_trad.train(show_train_summary=False)
        post_trad = inference_trad.build_posterior(de_trad, sample_with='direct')
        
        # 保存模型
        model_sub_dir = os.path.join(RESULTS_DIR, f"GPS_{gps}")
        if not os.path.exists(model_sub_dir): os.makedirs(model_sub_dir)
        
        torch.save(de_ai, os.path.join(model_sub_dir, f'model_ai_seed_{seed}.pth'))
        torch.save(de_trad, os.path.join(model_sub_dir, f'model_trad_seed_{seed}.pth'))
        
        # -----------------------------------
        # B. 评估阶段 (RELEASED SEED)
        # -----------------------------------
        release_seed() # <--- 关键：释放种子，确保测试随机
        
        TESTS_PER_ROUND = 50
        seg_len = 8192
        max_idx_test = len(test_h1) - seg_len
        
        ul_ai_list = []
        ul_trad_list = []
        
        # 执行随机测试
        for i in range(TESTS_PER_ROUND):
            # 随机切片 (True Random)
            start_idx = np.random.randint(0, max_idx_test)
            slice_h1 = test_h1[start_idx : start_idx + seg_len].copy()
            slice_l1 = test_l1[start_idx : start_idx + seg_len].copy()
            
            # 白化与特征提取
            slice_h1 = robust_whiten(slice_h1, fs=2048)
            slice_l1 = robust_whiten(slice_l1, fs=2048)
            
            cc, _ = pearsonr(slice_h1, slice_l1)
            k_h1 = np.log1p(np.abs(kurtosis(slice_h1)))
            k_l1 = np.log1p(np.abs(kurtosis(slice_l1)))
            p = np.log10(np.var(slice_h1) * np.var(slice_l1) + 1e-30)
            
            obs_full = torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32).to(device)
            obs_trad = torch.tensor([cc, p], dtype=torch.float32).to(device)
            
            s_ai = post_ai.sample((1000,), x=obs_full, show_progress_bars=False)
            ul_ai_val = np.percentile(s_ai.cpu().numpy()[:, 0], 95)
            ul_ai_list.append(ul_ai_val)
            
            s_trad = post_trad.sample((1000,), x=obs_trad, show_progress_bars=False)
            ul_trad_val = np.percentile(s_trad.cpu().numpy()[:, 0], 95)
            ul_trad_list.append(ul_trad_val)
            
        # 计算统计量
        mean_ai = np.mean(ul_ai_list)
        mean_trad = np.mean(ul_trad_list)
        
        # 画图并保存
        plt.figure(figsize=(8, 6))
        plt.hist(ul_ai_list, bins=15, density=True, alpha=0.5, color='blue', label=f'ING-Net (Seed {seed})')
        plt.hist(ul_trad_list, bins=15, density=True, alpha=0.5, color='orange', label='Traditional')
        plt.axvline(mean_ai, color='blue', linestyle='--', linewidth=2)
        plt.axvline(mean_trad, color='orange', linestyle='--', linewidth=2)
        plt.xlabel(r'95% Upper Limit ($\log_{10}\Omega$)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Evaluation: GPS {gps} | Seed {seed}\n(Random Test Slices)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        img_name = f"Eval_GPS_{gps}_Seed_{seed}.png"
        plt.savefig(os.path.join(model_sub_dir, img_name))
        plt.close() # <--- 关键：立即关闭，防止内存泄漏或卡住
        
        print(f"      [OK] Model Saved & Evaluated. (Plot: {img_name})")

print("\n==========================================================")
print(f"   所有任务完成！")
print(f"   共生成 100 个模型及对应的评价图表。")
print(f"   请查看文件夹: {RESULTS_DIR}")
print(f"==========================================================")
