"""
Author: Shannon | 2025.5.26
功能实现：
1. EEG数据加载和处理工具函数
2. 波形图和注意力水平可视化
3. 数据验证和辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS的中文字体
except:
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows的中文字体
    except:
        print("警告：未能找到合适的中文字体，标题可能显示不正确")
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_brainflow_data(file_path):
    """
    加载Brainflow格式的EEG数据
    
    参数:
        file_path (str): 数据文件路径
    返回:
        ndarray: EEG数据，形状为 (samples, channels)
    """
    data = np.loadtxt(file_path)
    # 提取EEG通道数据（第2-9列）
    eeg_data = data[:, 1:9]
    return eeg_data

def plot_eeg_channels(data, fs=500, duration=5, title="EEG Channels"):
    """
    绘制EEG通道数据
    
    参数:
        data (ndarray): EEG数据，形状为 (samples, channels)
        fs (int): 采样频率
        duration (float): 要显示的时间长度（秒）
        title (str): 图表标题
    """
    n_channels = data.shape[1]
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs
    
    plt.figure(figsize=(15, 10))
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(time, data[:n_samples, i])
        plt.ylabel(f'Channel {i+1}')
        if i == n_channels-1:
            plt.xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_frequency_response(lowcut, highcut, fs, order=4):
    """
    绘制带通滤波器的频率响应
    
    参数:
        lowcut (float): 低频截止频率
        highcut (float): 高频截止频率
        fs (float): 采样频率
        order (int): 滤波器阶数
    """
    from scipy.signal import freqz
    from preprocessing import butter_bandpass
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    
    plt.figure(figsize=(10, 4))
    plt.plot(0.5*fs*w/np.pi, np.abs(h))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid()
    plt.title(f'Bandpass Filter Frequency Response\n{lowcut}-{highcut} Hz')
    plt.axvline(lowcut, color='r')
    plt.axvline(highcut, color='r')
    plt.show()

def plot_filtered_comparison(raw_data, filtered_data, fs=500, duration=5):
    """
    对比显示原始数据和滤波后的数据
    
    参数:
        raw_data (ndarray): 原始EEG数据
        filtered_data (ndarray): 滤波后的EEG数据
        fs (int): 采样频率
        duration (float): 要显示的时间长度（秒）
    """
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs
    
    plt.figure(figsize=(15, 8))
    
    # 绘制第一个通道的对比
    plt.subplot(2, 1, 1)
    plt.plot(time, raw_data[:n_samples, 0], label='Raw')
    plt.title('Raw EEG - Channel 1')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_data[:n_samples, 0], label='Filtered')
    plt.title('Filtered EEG - Channel 1')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 