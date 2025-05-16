"""
Author: Shannon | 2025.5.26
功能实现：
1. 实现EEG数据预处理流程
2. 带通滤波（1-45Hz）实现
3. 数据分窗和标准化处理
"""

import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    设计Butterworth带通滤波器
    
    参数:
        lowcut (float): 低频截止频率
        highcut (float): 高频截止频率
        fs (float): 采样频率
        order (int): 滤波器阶数
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=250, order=4):
    """
    应用带通滤波器到EEG数据
    
    参数:
        data (ndarray): 形状为 (samples, channels) 的EEG数据
        lowcut (float): 低频截止频率
        highcut (float): 高频截止频率
        fs (float): 采样频率
        order (int): 滤波器阶数
    """
    filtered_data = np.zeros_like(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    
    for channel in range(data.shape[1]):
        filtered_data[:, channel] = lfilter(b, a, data[:, channel])
    return filtered_data

def extract_windows(data, window_size=250, stride=125):
    """
    使用滑动窗口提取数据片段
    
    参数:
        data (ndarray): 形状为 (samples, channels) 的EEG数据
        window_size (int): 窗口大小（样本数）
        stride (int): 滑动步长
    """
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    windows = []
    
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]
        if window.shape[0] == window_size:  # 确保窗口大小正确
            windows.append(window)
    
    return np.array(windows)

def scale_features(data):
    """
    标准化特征
    
    参数:
        data (ndarray): 形状为 (windows, samples, channels) 的EEG数据
    """
    # 对每个窗口分别进行标准化
    scaled_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            scaler = StandardScaler()
            scaled_data[i, :, j] = scaler.fit_transform(data[i, :, j].reshape(-1, 1)).ravel()
    return scaled_data

def preprocess_eeg(raw_data, window_size=250, stride=125):
    """
    完整的EEG预处理流程
    
    参数:
        raw_data (ndarray): 原始EEG数据，形状为 (samples, 23)
        window_size (int): 窗口大小（样本数）
        stride (int): 滑动步长
    """
    # 1. 提取EEG通道数据（第2-9列）
    eeg_data = raw_data[:, 1:9]
    
    # 2. 带通滤波
    filtered_data = apply_bandpass_filter(eeg_data)
    
    # 3. 窗口采样
    windowed_data = extract_windows(filtered_data, window_size, stride)
    
    # 4. 特征缩放
    processed_data = scale_features(windowed_data)
    
    # 5. 转换为模型所需的格式 (windows, channels, samples)
    processed_data = np.transpose(processed_data, (0, 2, 1))
    
    return processed_data 