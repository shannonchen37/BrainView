"""
Author: Shannon | 2025.5.26
功能实现：
1. EEGNet: 实现CNN用于EEG空间特征提取
2. Combine: 实现CNN+LSTM混合模型用于注意力分类
3. AttentionClassifier: 提供模型加载和预测接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 将原始模型目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../EEG/src')))

class EEGNet(nn.Module):
    def __init__(self, num_classes, combined, fc_size, dropout):
        super(EEGNet, self).__init__()
        self.T = 120
        self.combined = combined
        self.fc_size = fc_size
        self.dropout = dropout
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 8), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        self.fc1 = nn.Linear(fc_size, num_classes)
        
    def forward(self, x):
        # Layer 1
        x = x.float()
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dropout)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.reshape(x.size(0), -1)  # 使用reshape替代view
        if not self.combined:      
            x = self.fc1(x)
        return x

class Combine(nn.Module):
    def __init__(self, num_classes=2, combined=True, fc_size=56, dropout=0.9):
        super(Combine, self).__init__()
        self.cnn = EEGNet(num_classes, combined, fc_size, dropout)
        self.rnn = nn.LSTM(
            input_size=fc_size, 
            hidden_size=16, 
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(16, num_classes)

    def forward(self, x):
        batch_size, C, timepoints, channels = x.size()
        c_in = x
        c_out = self.cnn(c_in)
        r_in = c_out.reshape(batch_size, 1, -1)  # 使用reshape替代view
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        return F.log_softmax(r_out2, dim=1)

# 注册模型类到全局命名空间
sys.modules['__main__'].Combine = Combine
sys.modules['__main__'].EEGNet = EEGNet

class AttentionClassifier:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化注意力分类器
        
        参数:
            model_path: 预训练模型的路径
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model = Combine().to(device)
        
        if model_path is None:
            # 使用相对于attention_classifier目录的路径
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'Hybrid.pth')
        
        try:
            # 直接加载整个模型，关闭weights_only
            loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(loaded_model, dict):
                # 如果加载的是状态字典
                self.model.load_state_dict(loaded_model)
            else:
                # 如果加载的是整个模型
                self.model = loaded_model.to(device)
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"尝试加载路径: {model_path}")
            # 如果模型加载失败，初始化一个新模型
            print("初始化新模型...")
            self.model = Combine().to(device)
        
        self.model.eval()
    
    def predict(self, X):
        """
        使用预训练模型进行预测
        
        参数:
            X (ndarray): 形状为 (n_windows, n_channels, n_samples) 的EEG数据
            
        返回:
            ndarray: 预测的类别标签 (0: 低注意力, 1: 高注意力)
        """
        self.model.eval()
        # 调整输入格式为 (batch, 1, timepoints, channels)
        X = np.transpose(X, (0, 2, 1))  # 从 (batch, channels, timepoints) 到 (batch, timepoints, channels)
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])  # 添加通道维度
        X = torch.FloatTensor(X).contiguous().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        预测每个类别的概率
        
        参数:
            X (ndarray): 形状为 (n_windows, n_channels, n_samples) 的EEG数据
            
        返回:
            ndarray: 每个类别的概率 (shape: (n_windows, n_classes))
        """
        self.model.eval()
        # 调整输入格式为 (batch, 1, timepoints, channels)
        X = np.transpose(X, (0, 2, 1))
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        X = torch.FloatTensor(X).contiguous().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.exp(outputs)  # 将log_softmax转换回概率
        return probabilities.cpu().numpy() 