"""
Author: Shannon | 2025.5.26
功能实现：
1. 示例代码：展示完整的使用流程
2. 包含数据生成、预处理、预测和可视化
3. 作为项目使用的参考样例
"""

import numpy as np
import pandas as pd
import os
from preprocessing import preprocess_eeg
from model import AttentionClassifier
from utils import load_brainflow_data, plot_eeg_channels, plot_filtered_comparison
import matplotlib.pyplot as plt
from scipy import signal

def preprocess_eeg(raw_data, sfreq=500, window_size=120, window_step=60):
    """
    预处理EEG数据
    
    参数:
        raw_data: 形状为 (channels, samples) 的原始EEG数据
        sfreq: 采样频率
        window_size: 窗口大小（采样点数）
        window_step: 窗口步长（采样点数）
    
    返回:
        windows: 形状为 (n_windows, channels, window_size) 的预处理后的数据
    """
    # 1. 带通滤波 (1-45 Hz)
    nyq = sfreq / 2
    b, a = signal.butter(4, [1/nyq, 45/nyq], btype='bandpass')
    filtered_data = signal.filtfilt(b, a, raw_data, axis=1)
    
    # 2. 分窗
    n_channels, n_samples = filtered_data.shape
    n_windows = (n_samples - window_size) // window_step + 1
    windows = np.zeros((n_windows, n_channels, window_size))
    
    for i in range(n_windows):
        start = i * window_step
        end = start + window_size
        windows[i] = filtered_data[:, start:end]
    
    # 3. Z-score标准化
    for i in range(n_windows):
        for j in range(n_channels):
            windows[i, j] = (windows[i, j] - np.mean(windows[i, j])) / np.std(windows[i, j])
    
    return windows

def plot_eeg(data, sfreq=500, title="EEG数据", save_path=None):
    """绘制EEG数据"""
    n_channels = data.shape[0]
    time = np.arange(data.shape[1]) / sfreq
    
    plt.figure(figsize=(15, 8))
    for i in range(n_channels):
        plt.plot(time, data[i] + i*4, label=f'通道 {i+1}')
    
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅 + 偏移')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_attention_levels(predictions, probabilities, window_size=120, window_step=60, sfreq=500, save_path=None):
    """
    绘制注意力水平变化图
    
    参数:
        predictions: 每个窗口的预测结果（0或1）
        probabilities: 每个窗口的预测概率
        window_size: 窗口大小（采样点数）
        window_step: 窗口步长（采样点数）
        sfreq: 采样频率
        save_path: 保存图片的路径
    """
    # 计算每个窗口的中心时间点
    n_windows = len(predictions)
    window_centers = np.arange(n_windows) * window_step / sfreq + window_size / (2 * sfreq)
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制注意力水平（离散值）
    plt.subplot(2, 1, 1)
    plt.step(window_centers, predictions, 'b-', where='mid', label='注意力水平')
    plt.fill_between(window_centers, predictions, step='mid', alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.ylabel('注意力水平')
    plt.title('注意力水平变化 (0: 不集中, 1: 集中)')
    plt.grid(True)
    plt.legend()
    
    # 绘制预测概率
    plt.subplot(2, 1, 2)
    plt.plot(window_centers, probabilities[:, 1], 'r-', label='高注意力概率')
    plt.fill_between(window_centers, probabilities[:, 1], alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('时间 (秒)')
    plt.ylabel('概率')
    plt.title('注意力水平概率变化')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    # 创建输出目录
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 生成示例数据（模拟8通道EEG数据，10秒，500Hz采样率）
    print("加载EEG数据...")
    n_channels = 8
    duration = 10  # 秒
    sfreq = 500  # Hz
    t = np.arange(0, duration, 1/sfreq)
    
    # 生成模拟的EEG数据（每个通道使用不同频率的正弦波）
    raw_data = np.zeros((n_channels, len(t)))
    for i in range(n_channels):
        # 基础频率为8Hz，每个通道加1Hz
        freq = 8 + i
        raw_data[i] = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(len(t))
    
    # 2. 显示原始数据
    print("显示原始EEG数据...")
    plot_eeg(raw_data, sfreq, "原始EEG数据", os.path.join(output_dir, 'raw_eeg.png'))
    
    # 3. 预处理数据
    print("预处理数据...")
    processed_data = preprocess_eeg(raw_data)
    
    # 4. 加载模型并进行预测
    print("加载预训练模型并进行预测...")
    classifier = AttentionClassifier()
    
    # 对每个窗口进行预测
    predictions = classifier.predict(processed_data)
    probabilities = classifier.predict_proba(processed_data)
    
    # 5. 显示结果
    print("\n预测结果:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        attention = "高" if pred == 1 else "低"
        print(f"窗口 {i+1}: 注意力水平 = {attention} (概率: 低={prob[0]:.3f}, 高={prob[1]:.3f})")
    
    # 6. 绘制注意力水平变化图
    print("\n绘制注意力水平变化图...")
    plot_attention_levels(predictions, probabilities, save_path=os.path.join(output_dir, 'attention_levels.png'))

if __name__ == "__main__":
    main() 