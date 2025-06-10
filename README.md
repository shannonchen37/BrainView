# BrainView

## 项目演示

<video src="docs/demo.mov" controls width="100%"></video>

[无法播放时打开演示视频](docs/demo.mov)

## 项目简介

清华大学 iMoonLab 实验室脑电数据分析系统。
BrainView 面向 OpenBCI Cyton 采集设备，提供 EEG 实时采集、数据流稳定化、信号预处理、注意力状态预测、视频刺激展示、受试者信息管理，以及 MRI/fMRI/DTI NIfTI 影像导入与脑电状态关联展示。
本项目定位为实验软件平台，不输出医学诊断结论。MRI/fMRI/DTI 模块用于脑影像数据管理、预览和与 EEG 注意力状态的联合展示。

## 核心功能

- OpenBCI Cyton 设备连接与 8 通道 EEG 实时采集
- EEG 实时滤波、波形显示、数据流状态提示和原始数据保存
- 注意力状态自动预测，并显示状态变化和高注意力概率曲线
- 受试者信息录入、视频刺激播放和实验过程日志记录
- MRI/fMRI/DTI NIfTI 文件导入、切片预览和影像元数据显示
- EEG 注意力结果与脑影像数据的联合摘要展示

## 运行方式

请先确认本机已安装 Python 3.8 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/main.py
```

## 使用流程

1. 启动 BrainView 后，在界面左侧录入受试者信息。
2. 点击“打开视频”，选择本地 `.mp4`、`.avi` 或 `.mov` 视频作为实验刺激材料。
3. 连接 OpenBCI Cyton 设备后，在“脑电分析”菜单中点击“开始读取脑电信号”。
4. 采集过程中查看 8 通道 EEG 波形、注意力状态、概率曲线和数据流日志。
5. 如需查看脑影像数据，通过“导入医学数据”菜单导入 MRI、fMRI 或 DTI 的 `.nii` / `.nii.gz` 文件。
6. 通过“核磁数据分析”菜单查看当前影像数据与 EEG 注意力状态的联合摘要。
7. 实验结束后，再次点击“停止采集”结束 EEG 数据读取。

## 数据输出

- EEG 原始采集数据会保存到 `Recordings/Brainflow-RAW*.txt`
- 界面日志会显示设备连接、采集状态、数据流质量和保存状态
- 导入的 MRI/fMRI/DTI 数据会在界面中显示切片预览和基础元数据

## 输入文件支持

- 视频刺激：`.mp4`、`.avi`、`.mov`
- 脑影像数据：`.nii`、`.nii.gz`
- EEG 采集设备：OpenBCI Cyton

## 注意事项

- 未连接 OpenBCI 设备时，可以打开软件查看界面和导入影像数据，但无法开始 EEG 实时采集。
- 请保持 `attention_classifier/models/Hybrid.pth` 文件在默认位置，以便注意力分类模型正常加载。
- 本软件仅用于实验数据采集、展示和分析辅助，不应作为医学诊断依据。

## 技术组成

- Python / PySide6：桌面端图形界面、视频播放、菜单和对话框
- BrainFlow / PySerial：OpenBCI Cyton 数据采集与串口连接
- PyQtGraph：多通道 EEG 实时波形与注意力趋势展示
- PyTorch：CNN + LSTM 注意力模型推理
- NumPy / SciPy / scikit-learn：数据缓冲、滤波和标准化
- nibabel：MRI/fMRI/DTI NIfTI 文件读取

## 技术实现细节

### 1. OpenBCI 设备接入

通过 PySerial 扫描 `usbserial` 串口、向 Cyton 板卡发送停止流、软重置、默认通道配置、8 通道增益配置和 250 Hz 采样率指令，保证设备在采集前处于一致的初始化状态。

通过 BrainFlow `BoardShim(BoardIds.CYTON_BOARD, params)` 建立采集会话，并使用 `start_stream()` 启动数据流，将 OpenBCI 硬件数据统一接入到 Python 数据矩阵中。

### 2. 实时采集与数据流稳定化

通过 `get_board_data(chunk_size)` 消费式读取 BrainFlow 缓冲区，确保每个 EEG 样本只进入处理链路一次，避免重复读取最新窗口造成样本重叠。

通过读取 BrainFlow package number 通道并按 256 取模检查连续性，实时识别串口传输中的丢包、重复包和较大数据间隙。

对小于等于 8 个采样点的包间隙执行线性插值，对重复 package number 样本直接跳过，对较大间隙记录告警但不强行插值，以减少短时传输抖动对波形连续性的影响。

采集线程维护 `received_samples`、`dropped_samples`、`recovered_samples`、`duplicate_samples` 和 `large_gaps` 等统计信息，并通过界面日志展示数据流质量。

当设备超过 3 秒无数据时，系统会提示数据中断；连续无数据后会重新初始化设备、恢复数据流并重置包序号状态。

### 3. 显示缓冲与预测缓冲分离

实时 EEG 数据进入界面后会分为 `display_data` 和 `classification_data` 两条链路。显示链路用于滤波和波形刷新，预测链路用于模型输入和标准化，两者互不影响。

EEG 波形显示使用 5 秒长度的环形 `data_buffer`，即使输入数据块长度不固定，也能保持 8 通道波形稳定滚动展示。

注意力模型使用独立的 `PredictionBuffer`，将连续 EEG 采样整理为模型需要的固定长度预测窗口。

### 4. 预测缓冲机制

`PredictionBuffer` 内部维护形状为 `(8, 120)` 的缓冲矩阵，对应 8 通道 EEG 和模型需要的 120 个采样点。主界面每次收到采集线程发来的 `(samples, 8)` 数据块后，按样本逐行写入预测缓冲。

当 `write_index` 达到 120 时，当前窗口会被复制为一次模型输入。窗口生成后，系统将后 60 个采样点搬移到缓冲区前半段，并把写入位置重置为 60。下一批数据再写入 60 个新点后，就能形成下一个 120 点窗口。

这种“120 点窗口 + 60 点步长”的滚动缓冲实现了 50% 重叠的连续注意力预测。在 250 Hz 实时采样率下，120 点窗口约对应 0.48 秒上下文，60 点步长约每 0.24 秒输出一次预测结果。

系统将最近 10 秒预测结果保存在 `attention_history` 和 `probability_history` 中，用于展示注意力状态阶梯图和高注意力概率曲线。

### 5. EEG 预处理与注意力分析

通过 BrainFlow `remove_environmental_noise(..., FIFTY)` 去除 50 Hz 工频干扰，提高实时 EEG 波形的可读性。

通过 `detrend`、1 Hz 高通滤波和 1-50 Hz 带通滤波，减少基线漂移以及低频、高频干扰对实时观察的影响。

模型推理前会对每个通道执行 Z-score 标准化，并将数据整理为 `(1, channels, window_size)` 输入格式，保证实时采集数据与注意力模型输入格式一致。

注意力分类器采用 CNN + LSTM 结构：CNN 提取 EEG 空间特征，LSTM 提取短时间序列特征，最后由全连接层输出低注意力或高注意力分类结果。

界面同时展示离散注意力状态和高注意力概率曲线，并提供 0.5 阈值参考线，便于观察注意力状态变化趋势。

### 6. 图形化实验平台

BrainView 使用 PySide6 将受试者信息录入、视频刺激播放、OpenBCI 采集控制、EEG 波形监测、注意力结果展示和日志状态整合到一个主窗口中。

视频刺激播放基于 `QMediaPlayer` 和 `QVideoWidget`，支持本地 `.mp4`、`.avi` 和 `.mov` 文件，实现“视频任务展示 + EEG 实时监测”的同步实验界面。

原始 EEG 数据通过 `DataFilter.write_file` 按时间戳写入 `Recordings/Brainflow-RAW*.txt`，便于实验结束后复盘与分析。

### 7. MRI/fMRI/DTI 数据展示与联合摘要

通过 nibabel 读取 `.nii` 和 `.nii.gz` 文件，并对 3D 数据提取中间切片、对 4D 数据提取第一帧，实现 MRI/fMRI/DTI 数据在桌面界面中的快速预览。

系统会记录影像文件名、原始维度、展示切片、强度均值、标准差、最小值和最大值，方便用户查看影像数据的基础信息。

导入影像数据后，系统会将最近一次 EEG 注意力预测、预测概率、预测窗口数量和影像元数据合并为多模态摘要。

通过 MRI/fMRI/DTI 菜单可以生成当前模态与 EEG 状态的联合分析报告，报告包含影像元数据、注意力状态和数据流质量信息。
