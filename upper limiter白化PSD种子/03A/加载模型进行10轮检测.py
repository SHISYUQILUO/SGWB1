# @title Phase 12 (Evaluation): Load Seed 4040 Model & 10-Round Test
import os
import sys
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, pearsonr
import scipy.signal as signal
import re
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from tqdm import tqdm

# --- 1. 基础配置 ---
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# [配置区域] 路径设置
# ==========================================
# 1. 原始 O3a 数据路径 (请确保这些文件存在)
H1_FILE = r"C:\Users\20466\Desktop\upper limiter\ligo_o3b_data\O3a_H1_1243436468.pt"
L1_FILE = r"C:\Users\20466\Desktop\upper limiter\ligo_o3b_data\O3a_L1_1243436468.pt"

# 2. [关键] 已训练模型的文件夹路径 (你提供的路径)
MODEL_DIR = r"C:\Users\20466\Desktop\之前 - 副本\Results_O3a_Seed4040_Analysis\Models_Seed_4040_20260209"

# 3. 结果保存路径 (保存生成的10张图片)
RESULTS_DIR = os.path.join(MODEL_DIR, "Evaluation_10_Rounds_Output")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

print(f"[Phase 12 - Evaluation Only] 开始运行 | 设备: {device}")
print(f"模型加载路径: {MODEL_DIR}")
print(f"结果保存路径: {RESULTS_DIR}")

# ==========================================
# [核心] 鲁棒白化函数 (保持一致)
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
# 2. 数据准备 (加载测试集)
# ==========================================
print("\n>>> [1/4] 加载 O3a 数据文件...")

if not os.path.exists(H1_FILE) or not os.path.exists(L1_FILE):
    print(f"错误：未找到 O3a 数据文件！请检查路径:\n{H1_FILE}")
    sys.exit()

full_h1 = torch.load(H1_FILE, map_location='cpu')
full_l1 = torch.load(L1_FILE, map_location='cpu')

if isinstance(full_h1, torch.Tensor): full_h1 = full_h1.numpy().flatten()
if isinstance(full_l1, torch.Tensor): full_l1 = full_l1.numpy().flatten()

# 划分测试集 (后半部分)
min_len = min(len(full_h1), len(full_l1))
mid_point = min_len // 2
test_h1, test_l1 = full_h1[mid_point:], full_l1[mid_point:]
print(f"    测试集数据就绪: {len(test_h1)/2048:.1f}s")

# ==========================================
# 3. 加载模型
# ==========================================
print(f"\n>>> [2/4] 从指定文件夹加载 Seed 4040 模型...")

# 构造文件名 (基于之前的命名逻辑)
ai_model_path = os.path.join(MODEL_DIR, "model_ai_seed_4040.pth")
trad_model_path = os.path.join(MODEL_DIR, "model_trad_seed_4040.pth")

if not os.path.exists(ai_model_path):
    print(f"❌ 错误：在文件夹中找不到 {ai_model_path}")
    print("请检查文件夹内文件名是否为 'model_ai_seed_4040.pth'")
    sys.exit()

# 定义先验 (用于构建后验对象)
prior = BoxUniform(low=torch.tensor([-25.0, 0.001], device=device), 
                   high=torch.tensor([-5.0, 1.0], device=device))

try:
    # 1. 加载密度估计器 (Density Estimator)
    de_ai = torch.load(ai_model_path, map_location=device)
    de_trad = torch.load(trad_model_path, map_location=device)
    
    # 2. 重建后验对象 (Posterior)
    # 使用空的 SNPE 实例来构建后验
    inference_loader = SNPE(prior=prior, device=device)
    post_ai = inference_loader.build_posterior(de_ai, sample_with='direct')
    post_trad = inference_loader.build_posterior(de_trad, sample_with='direct')
    
    print("    ✅ 模型加载并重组成功！")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit()

# ==========================================
# 4. 执行 10 轮测试并绘图
# ==========================================
print(f"\n>>> [3/4] 开始 10 轮随机性测试与评价...")

NUM_ROUNDS = 10         # 修改：运行 10 次
TESTS_PER_ROUND = 50    # 每轮测试 50 个样本 (可根据速度调整)
seg_len = 8192
max_idx_test = len(test_h1) - seg_len

# 确保随机性 (释放种子)
np.random.seed(None)

for round_idx in range(1, NUM_ROUNDS + 1):
    print(f"\n--- [Round {round_idx}/{NUM_ROUNDS}] ---")
    
    ul_ai_list = []
    ul_trad_list = []
    
    # --- 批量测试 ---
    for i in tqdm(range(TESTS_PER_ROUND), desc=f"Testing Round {round_idx}", leave=False):
        # 1. 随机切片
        start_idx = np.random.randint(0, max_idx_test)
        slice_h1 = test_h1[start_idx : start_idx + seg_len].copy()
        slice_l1 = test_l1[start_idx : start_idx + seg_len].copy()
        
        # 2. 预处理
        slice_h1 = robust_whiten(slice_h1, fs=2048)
        slice_l1 = robust_whiten(slice_l1, fs=2048)
        
        # 3. 特征提取
        cc, _ = pearsonr(slice_h1, slice_l1)
        k_h1 = np.log1p(np.abs(kurtosis(slice_h1)))
        k_l1 = np.log1p(np.abs(kurtosis(slice_l1)))
        p = np.log10(np.var(slice_h1) * np.var(slice_l1) + 1e-30)
        
        obs_full = torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32).to(device)
        obs_trad = torch.tensor([cc, p], dtype=torch.float32).to(device)
        
        # 4. 推断 (Inference)
        # 采样 1000 个后验点，取 95% 分位数为上限
        s_ai = post_ai.sample((1000,), x=obs_full, show_progress_bars=False)
        ul_ai_val = np.percentile(s_ai.cpu().numpy()[:, 0], 95)
        ul_ai_list.append(ul_ai_val)
        
        s_trad = post_trad.sample((1000,), x=obs_trad, show_progress_bars=False)
        ul_trad_val = np.percentile(s_trad.cpu().numpy()[:, 0], 95)
        ul_trad_list.append(ul_trad_val)

    # --- 统计结果 ---
    mean_ai = np.mean(ul_ai_list)
    mean_trad = np.mean(ul_trad_list)
    improvement = 10**mean_trad / 10**mean_ai
    
    print(f"    > Round {round_idx} Result: Trad=10^{mean_trad:.2f} | AI=10^{mean_ai:.2f} | Improvement={improvement:.2f}x")

    # --- 绘图与保存 ---
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(ul_ai_list, bins=15, density=True, alpha=0.6, color='royalblue', label=f'ING-Net (Seed 4040)')
    plt.hist(ul_trad_list, bins=15, density=True, alpha=0.6, color='darkorange', label='Traditional Baseline')
    
    # 绘制均值线
    plt.axvline(mean_ai, color='blue', linestyle='--', linewidth=2, label=f'AI Mean: {mean_ai:.2f}')
    plt.axvline(mean_trad, color='darkorange', linestyle='--', linewidth=2, label=f'Trad Mean: {mean_trad:.2f}')
    
    # 图表装饰
    plt.xlabel(r'95% Upper Limit ($\log_{10}\Omega$)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Evaluation Round {round_idx}/10: Sensitivity Comparison\n(Improvement Factor: {improvement:.2f}x)', fontsize=15)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存图片
    save_filename = f"Eval_Round_{round_idx:02d}_Seed4040.png"
    save_path = os.path.join(RESULTS_DIR, save_filename)
    plt.savefig(save_path, dpi=300)
    plt.close() # 关闭画布，防止内存溢出
    
    print(f"    🖼️ 图片已保存: {save_path}")

print(f"\n{'='*60}")
print(f"🎉 全部 10 轮测试完成！")
print(f"📂 所有图片保存在: {RESULTS_DIR}")
print(f"{'='*60}")