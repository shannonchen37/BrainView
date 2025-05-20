# EEG注意力分类器

这个项目实现了一个基于EEG信号的注意力水平分类器。使用混合神经网络模型（CNN+LSTM）对EEG信号进行处理和分类，可以实时预测用户的注意力水平。

## 项目结构

```
attention_classifier/
├── models/             # 预训练模型
│   └── Hybrid.pth     # 预训练的混合模型
├── output/            # 输出文件夹
│   ├── raw_eeg.png    # EEG原始数据可视化
│   └── attention_levels.png  # 注意力水平变化图
├── example.py         # 示例代码
├── model.py           # 模型定义
├── preprocessing.py   # 数据预处理
├── utils.py          # 工具函数
└── requirements.txt   # 依赖包列表
```

## 安装

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 示例代码

运行示例代码以查看模型的工作方式：
```bash
python example.py
```

这将：
1. 生成模拟的8通道EEG数据
2. 显示并保存原始EEG数据的波形图
3. 对数据进行预处理（带通滤波、分窗、标准化）
4. 使用预训练模型进行注意力水平预测
5. 显示并保存注意力水平变化图

### 在自己的项目中使用

```python
from model import AttentionClassifier
import numpy as np

# 初始化分类器
classifier = AttentionClassifier()

# 准备数据 (batch_size, n_channels=8, n_samples=120)
eeg_data = np.random.randn(1, 8, 120).astype(np.float32)

# 获取预测结果
predictions = classifier.predict(eeg_data)  # 返回0（低注意力）或1（高注意力）

# 获取预测概率
probabilities = classifier.predict_proba(eeg_data)  # 返回每个类别的概率
```

## 数据预处理

预处理步骤包括：
1. 带通滤波（1-45 Hz）
2. 时间窗口划分（窗口大小=120采样点，步长=60采样点）
3. Z-score标准化

## 模型架构

使用混合神经网络模型，包含：
1. CNN部分：用于空间特征提取
2. LSTM部分：用于时序特征提取
3. 全连接层：用于最终分类

## 输出说明

1. 预测结果：
   - 0：低注意力
   - 1：高注意力

2. 可视化输出：
   - `raw_eeg.png`：显示原始EEG数据的波形
   - `attention_levels.png`：显示注意力水平变化和预测概率

## 注意事项

1. 输入数据要求：
   - 8通道EEG数据
   - 采样频率：500Hz
   - 数据格式：(batch_size, channels=8, samples=120)

2. 预处理参数：
   - 带通滤波范围：1-45Hz
   - 窗口大小：120采样点（0.24秒）
   - 窗口步长：60采样点（50%重叠）

## 引用

如果您在研究中使用了这个项目，请引用原论文：[论文引用信息]

## Example Data

The `example_eeg_data` directory contains the following example files:

1. `clean_trainingfiles.csv` - Cleaned training data file containing metadata about EEG recordings
2. `Trainingfiles.csv` - Original training data file with metadata
3. `All EEG files.csv` - Complete list of all EEG recordings
4. `example_raw_eeg.csv` - Example raw EEG data file with 8 channels

### Data Format

The raw EEG data file (`example_raw_eeg.csv`) contains:
- 8 EEG channels (C1-C8)
- Timestamp
- AdjustedUnix timestamp
- Sampling rate: 500Hz
- Data format: CSV with floating-point values

The metadata files contain information about:
- File paths
- User IDs
- Test types
- Number of paragraphs
- Data quality indicators
- Row counts 