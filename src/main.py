import sys
import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import traceback  # 添加traceback导入

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import torch
except ImportError:
    print("错误：未安装 PyTorch。请运行：pip install torch")
    sys.exit(1)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QVBoxLayout, QFileDialog, QHeaderView,
                               QPushButton, QHBoxLayout, QWidget, QLabel, QTableWidget, QLineEdit, QComboBox,
                               QFrame, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QTextEdit,
                               QStatusBar, QSizePolicy)
from PySide6.QtCore import Qt, QUrl, QDateTime, QTimer
from PySide6.QtGui import QFont, QTransform, QAction, QIntValidator, QImage, QPixmap, QIcon
import pyqtgraph as pg
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes
from EEGThread import EEGThread
import nibabel as nib

# Shannon | 2025.5.27 | 实现macOS系统字体自适应
if platform.system() == 'Darwin':  # macOS
    plt.rcParams["font.family"] = ["PingFang SC"]
    DEFAULT_FONT = "PingFang SC"
else:  # Windows 或其他系统
    plt.rcParams["font.family"] = ["SimHei"]
    DEFAULT_FONT = "Microsoft YaHei"


class PredictionBuffer:
    """将连续EEG采样整理成模型需要的滚动预测窗口。"""

    def __init__(self, num_channels=8, window_size=120, stride=60):
        self.num_channels = num_channels
        self.window_size = window_size
        self.stride = max(1, min(stride, window_size))
        self.buffer = np.zeros((num_channels, window_size))
        self.write_index = 0
        self.total_samples = 0

    def reset(self):
        self.buffer = np.zeros((self.num_channels, self.window_size))
        self.write_index = 0
        self.total_samples = 0

    def append_samples(self, samples):
        """追加(samples, channels)数据，返回已凑满的(channels, window_size)窗口。"""
        if samples.ndim != 2 or samples.shape[1] != self.num_channels:
            return []

        completed_windows = []
        for sample in samples:
            self.buffer[:, self.write_index] = sample
            self.write_index += 1
            self.total_samples += 1

            if self.write_index >= self.window_size:
                completed_windows.append(self.buffer.copy())

                overlap = self.window_size - self.stride
                if overlap > 0:
                    self.buffer[:, :overlap] = self.buffer[:, self.stride:self.window_size]
                    self.write_index = overlap
                else:
                    self.write_index = 0

        return completed_windows

    @property
    def fill_ratio(self):
        return min(self.write_index / self.window_size, 1.0)


class MedicalDataVisualization(QMainWindow):
    def __init__(self):
        super().__init__()

        self.set_fonts()
        self.setWindowTitle("清华大学iMoonLab-脑电数据分析系统")
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen().geometry()
        # 设置窗口大小为屏幕高度的80%，宽度的75%
        window_height = int(screen.height() * 0.8)
        window_width = int(screen.width() * 0.75)
        # 计算窗口位置，使其居中显示
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        self.setMinimumSize(900, 560)
        self.setWindowIcon(QIcon("resources/images/icon.png"))

        # 初始化状态栏
        self.statusBar = self.statusBar()
        self.statusBar.setStyleSheet("""
            QStatusBar {
                background-color: #f0f0f0;
                color: #333333;
                border-top: 1px solid #cccccc;
                min-height: 20px;
                padding: 2px;
            }
        """)
        
        # 创建日志显示区域
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(48)
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                min-height: 60px;
            }
        """)
        self.log_layout.addWidget(self.log_text)
        self.log_layout.setContentsMargins(10, 0, 10, 10)  # 设置边距

        self.patient_data = []
        self.current_row = 0
        self.raw_data_labels = {}
        self.graphics_views = {}
        self.eeg_thread = None  # 初始化 EEG 线程
        self.eeg_period_sampling = False  # 初始化 EEG 周期采样标志
        self.save_dir = None  # 初始化保存目录

        # 初始化参数
        self.sample_rate = 250  # 采样率250Hz
        self.display_duration = 5  # 显示5秒数据
        self.display_samples = self.sample_rate * self.display_duration
        self.num_channels = 8  # 8通道
        self.prediction_window_size = 120
        self.prediction_stride = 60
        self.attention_history_seconds = 10
        self.prediction_buffer = PredictionBuffer(
            num_channels=self.num_channels,
            window_size=self.prediction_window_size,
            stride=self.prediction_stride
        )
        self.latest_stream_stats = {}
        self.eeg_summary = {
            'prediction': None,
            'probability': None,
            'timestamp': None,
            'total_windows': 0
        }
        self.medical_data = {}
        self.analysis_reports = []

        # 初始化注意力分类器
        try:
            from attention_classifier.model import AttentionClassifier
            self.attention_classifier = AttentionClassifier(device='cpu')  # 强制使用CPU设备
            self.log_message("注意力分类器初始化成功", "info")
        except Exception as e:
            self.log_message(f"注意力分类器初始化失败: {str(e)}", "error")
            self.attention_classifier = None
        
        # 初始化数据缓冲区
        self.eeg_buffer = np.zeros((self.num_channels, 120))  # 120个采样点，对应模型要求
        self.buffer_index = 0
        
        # 初始化结果缓冲区（存储10秒的预测结果）
        history_length = int(np.ceil(self.attention_history_seconds * self.sample_rate / self.prediction_stride))
        self.attention_history = np.zeros(history_length)  # 存储10秒的预测结果
        self.probability_history = np.zeros(history_length)  # 存储注意力概率
        self.time_points = np.linspace(-self.attention_history_seconds, 0, history_length)

        self.create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)

        upper_area = self.create_upper_area()
        main_layout.addWidget(upper_area, stretch=8)

        lower_area = self.create_lower_area()
        main_layout.addWidget(lower_area, stretch=2)

        self.multimodal_summary_label = QLabel("多模态关联：等待导入影像数据与EEG预测结果")
        self.multimodal_summary_label.setAlignment(Qt.AlignCenter)
        self.multimodal_summary_label.setWordWrap(True)
        self.multimodal_summary_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 6px;
                padding: 6px;
                color: #444444;
                font-size: 12px;
            }
        """)
        self.multimodal_summary_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout.addWidget(self.multimodal_summary_label, stretch=0)

        # 添加日志显示区域到主布局
        self.log_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.log_widget, stretch=0)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f9f9f9;
            }
            QLabel {
                color: #333333;
                font-family: "%s";
            }
            QPushButton {
                background-color: #9c27b0;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-family: "%s";
            }
            QPushButton:hover {
                background-color: #ab47bc;
            }
            QPushButton:pressed {
                background-color: #8e24aa;
            }
            QTableWidget {
                gridline-color: #dddddd;
                selection-background-color: #bde0fe;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 6px;
                border: 1px solid #cccccc;
                font-weight: bold;
                font-family: "%s";
            }
            QMenuBar {
                background-color: #f0f0f0;
                font-family: "%s";
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
            }
            QMenu {
                font-family: "%s";
            }
            QMenu::item:selected {
                background-color: #e0e0e0;
            }
            .DataGroup {
                border: 1px solid #cccccc;
                border-radius: 8px;
                background-color: #ffffff;
                padding: 10px;
                margin: 5px;
            }
            QGraphicsView {
                border: 1px solid #cccccc;
                border-radius: 6px;
                background-color: #f5f5f5;
            }
            QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
                background-color: #ffffff;
                font-family: "%s";
            }
        """ % (DEFAULT_FONT, DEFAULT_FONT, DEFAULT_FONT, DEFAULT_FONT, DEFAULT_FONT, DEFAULT_FONT))

    def set_fonts(self):
        font = QFont()
        font.setFamily(DEFAULT_FONT)
        font.setStyleStrategy(QFont.PreferAntialias)
        self.setFont(font)

    def create_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
                font-size: 14px;
            }
            QMenuBar::item {
                padding: 8px 16px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #e9ecef;
                border-radius: 4px;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 0;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #e9ecef;
            }
        """)

        import_menu = menubar.addMenu("导入医学数据")
        mri_action = QAction("MRI", self)
        mri_action.triggered.connect(lambda: self.import_data("MRI"))
        import_menu.addAction(mri_action)
        fmri_action = QAction("fMRI", self)
        fmri_action.triggered.connect(lambda: self.import_data("fMRI"))
        import_menu.addAction(fmri_action)
        dti_action = QAction("DTI", self)
        dti_action.triggered.connect(lambda: self.import_data("DTI"))
        import_menu.addAction(dti_action)

        eeg_menu = menubar.addMenu("脑电分析")
        self.action_EEG = QAction("开始读取脑电信号", self)
        self.action_EEG.triggered.connect(self.start_eeg_acquisition)
        eeg_menu.addAction(self.action_EEG)

        mri_analysis_menu = menubar.addMenu("核磁数据分析")
        mri_model_action = QAction("MRI模型", self)
        mri_model_action.triggered.connect(lambda: self.run_mri_model("MRI"))
        mri_analysis_menu.addAction(mri_model_action)
        fmri_model_action = QAction("fMRI模型", self)
        fmri_model_action.triggered.connect(lambda: self.run_mri_model("fMRI"))
        mri_analysis_menu.addAction(fmri_model_action)
        dti_model_action = QAction("DTI模型", self)
        dti_model_action.triggered.connect(lambda: self.run_mri_model("DTI"))
        mri_analysis_menu.addAction(dti_model_action)

    def create_upper_area(self):
        upper_widget = QWidget()
        upper_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        upper_layout = QHBoxLayout(upper_widget)
        left_area = self.create_left_area()
        upper_layout.addWidget(left_area, stretch=2)
        middle_area = self.create_middle_area()
        upper_layout.addWidget(middle_area, stretch=3)
        right_area = self.create_right_area()
        upper_layout.addWidget(right_area, stretch=3)
        return upper_widget

    def create_left_area(self):
        left_widget = QWidget()
        left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        title_label = QLabel("个人信息")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        left_layout.addWidget(title_label)

        self.patient_table = QTableWidget(1, 4)
        self.patient_table.setHorizontalHeaderLabels(["姓名", "年龄", "性别", "单位"])
        self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.patient_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                color: #333333;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                color: #333333;
            }
        """)

        self.name_edit = QLineEdit()
        self.name_edit.setAlignment(Qt.AlignCenter)
        self.patient_table.setCellWidget(0, 0, self.name_edit)

        self.age_edit = QLineEdit()
        self.age_edit.setValidator(QIntValidator(0, 150))
        self.age_edit.setAlignment(Qt.AlignCenter)
        self.patient_table.setCellWidget(0, 1, self.age_edit)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["男", "女"])
        self.gender_combo.setEditable(True)
        self.gender_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.gender_combo.setEditable(False)
        self.patient_table.setCellWidget(0, 2, self.gender_combo)

        self.unit_edit = QLineEdit()
        self.unit_edit.setAlignment(Qt.AlignCenter)
        self.patient_table.setCellWidget(0, 3, self.unit_edit)

        left_layout.addWidget(self.patient_table)

        button_layout = QHBoxLayout()
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self.clear_form)
        button_layout.addWidget(clear_button)
        add_button = QPushButton("添加")
        add_button.clicked.connect(self.add_patient)
        button_layout.addWidget(add_button)
        left_layout.addLayout(button_layout)

        self.patient_display = QWidget()
        self.patient_display_layout = QVBoxLayout(self.patient_display)
        self.patient_display_layout.setAlignment(Qt.AlignTop)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        id_header = QLabel("序号")
        id_header.setMinimumWidth(28)
        id_header.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(id_header, stretch=1)
        name_header = QLabel("姓名")
        name_header.setMinimumWidth(56)
        name_header.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(name_header, stretch=3)
        age_header = QLabel("年龄")
        age_header.setMinimumWidth(42)
        age_header.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(age_header, stretch=2)
        gender_header = QLabel("性别")
        gender_header.setMinimumWidth(42)
        gender_header.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(gender_header, stretch=2)
        unit_header = QLabel("单位")
        header_layout.addWidget(unit_header, stretch=4)
        self.patient_display_layout.addWidget(header_widget)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.patient_display_layout.addWidget(line)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.patient_display)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_layout.addWidget(scroll_area, stretch=1)

        return left_widget

    def create_middle_area(self):
        middle_widget = QWidget()
        middle_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        middle_layout = QVBoxLayout(middle_widget)

        title_label = QLabel("观看视频进行脑电数据分析")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        middle_layout.addWidget(title_label)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(240, 160)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        middle_layout.addWidget(self.video_widget, stretch=8)

        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.playbackStateChanged.connect(self.update_play_button)
        self.media_player.mediaStatusChanged.connect(self.toggle_playback)
        if hasattr(self.media_player, 'errorOccurred'):
            self.media_player.errorOccurred.connect(self.handle_video_error)
        else:
            self.media_player.error.connect(self.handle_video_error)

        button_layout = QHBoxLayout()
        open_button = QPushButton("打开视频")
        open_button.clicked.connect(self.open_video)
        button_layout.addWidget(open_button)
        self.play_button = QPushButton("播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        button_layout.addWidget(self.play_button)
        middle_layout.addLayout(button_layout)

        return middle_widget

    def create_right_area(self):
        right_widget = QWidget()
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_widget)

        # 标题
        eeg_label = QLabel("脑电信号接收与展示")
        eeg_label.setAlignment(Qt.AlignCenter)
        eeg_label.setFont(QFont(DEFAULT_FONT, 12, QFont.Bold))
        right_layout.addWidget(eeg_label)

        self.stream_status_label = QLabel("数据流：等待采集")
        self.stream_status_label.setAlignment(Qt.AlignCenter)
        self.stream_status_label.setWordWrap(True)
        self.stream_status_label.setStyleSheet("color: #555555; font-size: 12px;")
        right_layout.addWidget(self.stream_status_label)

        # 添加绘图窗口
        self.eeg_plot_widget = QWidget()
        self.eeg_plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._init_eeg_plot()
        right_layout.addWidget(self.eeg_plot_widget, stretch=3)

        # 注意力分析结果标题
        result_label = QLabel("注意力集中分析结果")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setFont(QFont(DEFAULT_FONT, 12, QFont.Bold))
        right_layout.addWidget(result_label)

        # 创建注意力显示区域
        attention_widget = QWidget()
        attention_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        attention_layout = QVBoxLayout(attention_widget)
        attention_layout.setContentsMargins(0, 0, 0, 0)
        attention_layout.setSpacing(4)
        
        # Shannon | 2025.5.27 | 实现注意力水平阶梯图显示
        self.attention_level_plot = pg.PlotWidget()
        self.attention_level_plot.setBackground('w')
        self.attention_level_plot.setMinimumHeight(24)
        self.attention_level_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.attention_level_plot.setXRange(-10, 0)  # 显示最近10秒的数据
        self.attention_level_plot.setYRange(-0.1, 1.1)
        self.attention_level_plot.getPlotItem().getAxis('left').setTicks([[(0, '不集中'), (1, '集中')]])
        self.attention_level_plot.getPlotItem().getAxis('bottom').setLabel('时间 (秒)')
        self.attention_level_curve = self.attention_level_plot.plot(stepMode='center', pen=pg.mkPen(color='b', width=2))
        attention_layout.addWidget(self.attention_level_plot, stretch=1)

        # Shannon | 2025.5.27 | 实现注意力概率实时曲线图
        self.attention_prob_plot = pg.PlotWidget()
        self.attention_prob_plot.setBackground('w')
        self.attention_prob_plot.setMinimumHeight(24)
        self.attention_prob_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.attention_prob_plot.setXRange(-10, 0)  # 显示最近10秒的数据
        self.attention_prob_plot.setYRange(-0.1, 1.1)
        self.attention_prob_plot.getPlotItem().getAxis('left').setLabel('概率')
        self.attention_prob_plot.getPlotItem().getAxis('bottom').setLabel('时间 (秒)')
        
        # Shannon | 2025.5.27 | 添加0.5阈值参考线
        threshold_line = pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen(color='#CCCCCC', width=1, style=Qt.DashLine))
        self.attention_prob_plot.addItem(threshold_line)
        
        self.attention_prob_curve = self.attention_prob_plot.plot(pen=pg.mkPen(color='r', width=2))
        attention_layout.addWidget(self.attention_prob_plot, stretch=1)

        right_layout.addWidget(attention_widget, stretch=3)

        return right_widget

    def create_lower_area(self):
        lower_widget = QWidget()
        lower_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lower_layout = QHBoxLayout(lower_widget)

        mri_group = self.create_data_group("MRI")
        lower_layout.addWidget(mri_group, stretch=1)
        fmri_group = self.create_data_group("fMRI")
        lower_layout.addWidget(fmri_group, stretch=1)
        dti_group = self.create_data_group("DTI")
        lower_layout.addWidget(dti_group, stretch=1)

        return lower_widget

    def create_data_group(self, title):
        group_widget = QWidget()
        group_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        group_widget.setObjectName("DataGroup")
        group_layout = QVBoxLayout(group_widget)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        group_layout.addWidget(title_label)

        data_layout = QHBoxLayout()
        self.raw_data_labels[title] = QLabel("原始数据")
        self.raw_data_labels[title].setAlignment(Qt.AlignCenter)
        self.raw_data_labels[title].setWordWrap(True)
        self.raw_data_labels[title].setMinimumSize(80, 80)
        self.raw_data_labels[title].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.raw_data_labels[title].setStyleSheet(
            "background-color: #ffffff; border: 1px solid #cccccc; border-radius: 6px;")
        data_layout.addWidget(self.raw_data_labels[title], stretch=2)

        self.graphics_views[title] = QGraphicsView()
        self.graphics_views[title].setMinimumSize(80, 80)
        self.graphics_views[title].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        data_layout.addWidget(self.graphics_views[title], stretch=3)

        group_layout.addLayout(data_layout, stretch=1)
        return group_widget

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refit_medical_views)

    def refit_medical_views(self):
        """窗口尺寸变化后按当前视图比例重新适配影像切片。"""
        for view in getattr(self, 'graphics_views', {}).values():
            scene = view.scene()
            if scene is not None and not scene.sceneRect().isNull():
                view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开视频文件", "",
                                                   "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)")
        if file_name:
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
            self.media_player.play()
            self.play_button.setEnabled(True)

    def toggle_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def update_play_button(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("暂停")
        else:
            self.play_button.setText("播放")

    def handle_video_error(self, error=None):
        error_string = self.media_player.errorString()
        QMessageBox.warning(self, "视频播放错误", f"播放视频时出错: {error_string}")
        self.play_button.setText("播放")
        self.play_button.setEnabled(False)

    def add_patient(self):
        name = self.name_edit.text()
        age = self.age_edit.text()
        gender = self.gender_combo.currentText()
        unit = self.unit_edit.text()
        if not name:
            QMessageBox.warning(self, "警告", "姓名不能为空")
            return
        self.current_row += 1
        patient_info = {
            "id": self.current_row,
            "name": name,
            "age": age,
            "gender": gender,
            "unit": unit
        }
        self.patient_data.append(patient_info)
        self.display_patient(patient_info)

    def display_patient(self, patient):
        patient_widget = QWidget()
        patient_layout = QHBoxLayout(patient_widget)
        id_label = QLabel(f"{patient['id']}.")
        id_label.setMinimumWidth(28)
        patient_layout.addWidget(id_label, stretch=1)
        name_label = QLabel(patient['name'])
        name_label.setMinimumWidth(56)
        patient_layout.addWidget(name_label, stretch=3)
        age_label = QLabel(patient['age'])
        age_label.setMinimumWidth(42)
        patient_layout.addWidget(age_label, stretch=2)
        gender_label = QLabel(patient['gender'])
        gender_label.setMinimumWidth(42)
        patient_layout.addWidget(gender_label, stretch=2)
        unit_label = QLabel(patient['unit'])
        patient_layout.addWidget(unit_label, stretch=4)
        self.patient_display_layout.addWidget(patient_widget)

    def clear_form(self):
        self.name_edit.clear()
        self.age_edit.clear()
        self.gender_combo.setCurrentIndex(0)
        self.unit_edit.clear()

    def import_data(self, data_type):
        file_path, _ = QFileDialog.getOpenFileName(self, f"导入{data_type}数据", "",
                                                   "NIfTI文件 (*.nii *.nii.gz);;所有文件 (*)")
        if file_path:
            try:
                self.display_nifti_image(file_path, data_type)
                # QMessageBox.information(self, "成功", f"{data_type}数据导入成功")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法解析文件: {str(e)}")

    def display_nifti_image(self, file_path, dicom_type):
        img = nib.load(file_path)
        data = img.get_fdata()
        original_shape = data.shape
        frame_note = ""
        if data.ndim == 4:
            data = data[:, :, :, 0]
            frame_note = "，4D数据已提取第一帧"
            self.log_message(f"检测到{dicom_type} 4D数据，提取第一帧用于展示", "info")
        slice_index = data.shape[2] // 2
        image_data = data[:, :, slice_index]
        min_value = float(np.min(image_data))
        max_value = float(np.max(image_data))
        value_range = max_value - min_value
        if value_range < 1e-8:
            image_data = np.zeros_like(image_data)
        else:
            image_data = (image_data - min_value) / value_range * 255
        image_data = image_data.astype(np.uint8)
        image_data = np.ascontiguousarray(image_data)
        height, width = image_data.shape
        q_image = QImage(image_data.data, width, height, width, QImage.Format_Grayscale8)
        if dicom_type in self.graphics_views:
            view = self.graphics_views[dicom_type]
            pixmap = QPixmap.fromImage(q_image)
            transform = QTransform().rotate(-90)
            pixmap = pixmap.transformed(transform)
            item = QGraphicsPixmapItem(pixmap)
            scene = QGraphicsScene()
            scene.addItem(item)
            view.setScene(scene)
            view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.medical_data[dicom_type] = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'shape': original_shape,
                'display_shape': data.shape,
                'slice_index': slice_index,
                'mean_intensity': float(np.mean(data)),
                'std_intensity': float(np.std(data)),
                'min_intensity': float(np.min(data)),
                'max_intensity': float(np.max(data)),
                'loaded_at': QDateTime.currentDateTime().toString("HH:mm:ss")
            }
            self.raw_data_labels[dicom_type].setText(
                f"{dicom_type}原始数据\n已加载{frame_note}\n\n"
                f"文件: {os.path.basename(file_path)}\n"
                f"维度: {original_shape}\n"
                f"切片: {slice_index}"
            )
            self.update_multimodal_summary()

    def update_stream_status(self, stats):
        """刷新UI中的数据流质量摘要。"""
        if not stats or not hasattr(self, 'stream_status_label'):
            return
        self.stream_status_label.setText(
            "数据流："
            f"接收 {stats.get('received_samples', 0)} 点 | "
            f"插值补点 {stats.get('recovered_samples', 0)} 点 | "
            f"重复 {stats.get('duplicate_samples', 0)} 点 | "
            f"大间隙 {stats.get('large_gaps', 0)} 次"
        )

    def format_attention_summary(self):
        """返回最近一次注意力预测的人类可读摘要。"""
        prediction = self.eeg_summary.get('prediction')
        probability = self.eeg_summary.get('probability')
        if prediction is None or probability is None:
            return "EEG注意力：等待预测结果"
        label = "集中" if prediction == 1 else "不集中"
        return (
            f"EEG注意力：{label} "
            f"(集中概率 {probability:.2f}, "
            f"窗口 {self.eeg_summary.get('total_windows', 0)} 个, "
            f"更新时间 {self.eeg_summary.get('timestamp')})"
        )

    def update_multimodal_summary(self):
        """将影像导入状态和最近EEG预测挂接成同一实验记录。"""
        if not hasattr(self, 'multimodal_summary_label'):
            return
        loaded_types = sorted(self.medical_data.keys())
        if loaded_types:
            image_summary = "影像数据：" + "、".join(
                f"{data_type}({self.medical_data[data_type]['file_name']})"
                for data_type in loaded_types
            )
        else:
            image_summary = "影像数据：等待导入"

        self.multimodal_summary_label.setText(
            f"多模态关联：{image_summary} | {self.format_attention_summary()}"
        )

    def build_multimodal_report(self, model_type):
        """生成当前模态与EEG注意力状态的联合分析摘要。"""
        image_info = self.medical_data.get(model_type)
        if image_info is None:
            return f"尚未导入{model_type}数据，请先通过“导入医学数据”菜单加载NIfTI文件。"

        stats = self.latest_stream_stats or {}
        report = [
            f"{model_type} - EEG联合分析摘要",
            f"影像文件: {image_info['file_name']}",
            f"影像维度: {image_info['shape']}",
            f"展示切片: {image_info['slice_index']}",
            f"强度均值/标准差: {image_info['mean_intensity']:.3f} / {image_info['std_intensity']:.3f}",
            self.format_attention_summary(),
            (
                "数据流质量: "
                f"接收{stats.get('received_samples', 0)}点, "
                f"插值补点{stats.get('recovered_samples', 0)}点, "
                f"重复{stats.get('duplicate_samples', 0)}点, "
                f"大间隙{stats.get('large_gaps', 0)}次"
            ),
            "结果: 已将该影像模态与当前受试者的EEG注意力状态关联记录，可用于后续结构/功能影像与脑电状态的联合分析。"
        ]
        return "\n".join(report)

    def _init_eeg_plot(self):
        """初始化EEG绘图界面
        设置绘图布局、通道显示和标签等
        """
        # Shannon | 2025.5.25 | 优化：初始化UI组件列表
        self.plot_widgets = []    # 存储每个通道的绘图窗口
        self.curves = []         # 存储每个通道的曲线对象
        self.uVrms_labels = []   # 存储每个通道的RMS值标签
        
        # Shannon | 2025.5.25 | 优化：定义通道颜色方案，提高可读性
        channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Shannon | 2025.5.25 | 优化：定义10-20系统电极位置名称
        channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        
        # 设置布局
        layout = QVBoxLayout(self.eeg_plot_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)  # 设置通道之间的间距为1像素
        
        # 设置基本参数
        self.sample_rate = 250  # 采样率250Hz
        self.display_duration = 5  # 显示5秒数据
        self.display_samples = self.sample_rate * self.display_duration
        self.current_index = 0
        self.num_channels = 8
        
        # 为每个通道创建独立的绘图窗口
        for i in range(self.num_channels):
            # 创建绘图窗口
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.setMinimumHeight(16 if i < self.num_channels - 1 else 24)
            plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # 设置X轴范围和样式
            plot_widget.setXRange(-self.display_duration, 0)
            plot_widget.getPlotItem().getAxis('bottom').setStyle(showValues=True if i == self.num_channels-1 else False)
            if i == self.num_channels-1:
                plot_widget.getPlotItem().getAxis('bottom').setLabel('Time (s)')
            
            # 设置Y轴范围和样式
            plot_widget.setYRange(-100, 100)  # 设置适当的Y轴范围
            plot_widget.getPlotItem().getAxis('left').setWidth(60)
            # Shannon | 2025.5.25 | 优化：使用10-20系统电极位置名称作为标签
            plot_widget.getPlotItem().getAxis('left').setLabel(channel_names[i])
            
            # 添加网格
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # 创建曲线
            pen = pg.mkPen(color=channel_colors[i], width=1)
            curve = plot_widget.plot(pen=pen)
            
            # 添加到布局
            layout.addWidget(plot_widget, stretch=1)
            
            # Shannon | 2025.5.25 | 优化：统一管理UI组件引用
            self.plot_widgets.append(plot_widget)
            self.curves.append(curve)

            # Shannon | 2025.5.25 | 优化：添加RMS值显示标签
            label = pg.TextItem("0.00 uVrms", anchor=(1, 0.5), color=channel_colors[i])
            plot_widget.addItem(label)
            label.setPos(0, 80)  # 位置可以调整
            self.uVrms_labels.append(label)

    def reset_eeg_analysis_state(self):
        """开始一轮新实验前清空显示缓冲、预测缓冲和结果历史。"""
        self.prediction_buffer.reset()
        self.eeg_buffer = np.zeros((self.num_channels, self.prediction_window_size))
        self.buffer_index = 0
        self.attention_history = np.zeros_like(self.attention_history)
        self.probability_history = np.zeros_like(self.probability_history)
        self.eeg_summary = {
            'prediction': None,
            'probability': None,
            'timestamp': None,
            'total_windows': 0
        }
        if hasattr(self, 'data_buffer'):
            self.data_buffer = np.zeros((self.num_channels, self.display_samples))
            self.current_index = 0
        if hasattr(self, 'stream_status_label'):
            self.stream_status_label.setText("数据流：等待采集")
        self.update_multimodal_summary()

    def start_eeg_acquisition(self):
        try:
            if not self.eeg_thread or not self.eeg_thread.isRunning():
                self.log_message("正在尝试启动EEG采集...", "info")
                self.reset_eeg_analysis_state()
                self.eeg_thread = EEGThread()
                
                # 使用新的消息框样式
                msg_box = self.create_message_box(
                    "连接状态",
                    "正在尝试连接EEG设备，请稍候...",
                    QMessageBox.Information
                )
                msg_box.exec_()
                
                # 添加信号连接的调试信息
                self.log_message("正在设置信号连接...", "info")
                self.eeg_thread.data_updated.connect(self.update_plot)
                self.eeg_thread.error_occurred.connect(self.show_error)
                self.eeg_thread.activated.connect(lambda: self.action_EEG.setText("停止采集"))
                self.eeg_thread.new_savefile.connect(self.new_savefile)
                self.eeg_thread.file_save.connect(self.save_file)
                self.eeg_thread.log_message.connect(lambda msg, level: self.log_message(msg, level, "OpenBCI"))
                
                self.log_message("启动EEG线程...", "info")
                self.eeg_thread.start()
                self.log_message("EEG线程已启动", "info")
                
            else:
                self.log_message("正在停止EEG采集...", "info")
                if self.eeg_period_sampling:
                    QMessageBox.warning(self, "强制结束时间采样", "注意，您正在强制结束时间采样")
                    self.eeg_period_sampling = False
                    if hasattr(self.eeg_thread, 'data_updated'):
                        try:
                            self.eeg_thread.data_updated.disconnect(self.record_data)
                        except AttributeError:
                            self.log_message("无法断开record_data连接", "warning")
                self.eeg_thread.running = False
                self.eeg_thread.wait()
                self.action_EEG.setText("开始读取脑电信号")
                self.log_message("EEG采集已停止", "info")
                
        except Exception as e:
            error_msg = f"启动EEG采集失败: {str(e)}"
            self.log_message(error_msg, "error")
            self.log_message(traceback.format_exc(), "error")
            QMessageBox.critical(self, "错误", error_msg)

    def update_plot(self, data_dict):
        """
        Shannon | 2025.5.27 | 实现实时EEG数据处理与显示
        功能：
        1. 对显示数据进行三重滤波（工频、基线漂移、带通）
        2. 保持原始数据用于注意力分类
        3. 实时更新波形显示和注意力预测
        """
        try:
            # 获取数据并进行基础检查
            data = data_dict['data']  # 已经是 (samples, channels) 格式
            sample_rate = data_dict['sample_rate']
            self.latest_stream_stats = data_dict.get('stream_stats', self.latest_stream_stats)
            self.update_stream_status(self.latest_stream_stats)
            
            # 数据格式验证
            if data.shape[1] != 8:
                self.log_message(f"数据通道数不正确: {data.shape[1]}", "warning")
                return
                
            # Shannon | 2025.5.27 | 实现数据分流处理
            # 创建数据副本，分别用于显示和分类
            display_data = data.copy()
            classification_data = data.copy()
            
            # Shannon | 2025.5.27 | 实现三重滤波
            try:
                # 对每个通道进行滤波
                for ch in range(8):
                    # 获取当前通道数据
                    channel_data = np.ascontiguousarray(display_data[:, ch])
                    
                    # 1. 工频滤波：去除50Hz工频干扰
                    DataFilter.remove_environmental_noise(
                        channel_data,
                        sample_rate,
                        NoiseTypes.FIFTY.value  # 使用枚举值指定50Hz工频
                    )
                    
                    # 2. 基线漂移校正：去除缓慢的基线漂移
                    DataFilter.detrend(channel_data, DetrendOperations.LINEAR.value)
                    DataFilter.perform_highpass(
                        channel_data,
                        sample_rate,
                        1.0,   # 截止频率1Hz
                        4,     # 4阶巴特沃斯滤波器
                        FilterTypes.BUTTERWORTH.value,
                        0
                    )
                    
                    # 3. 带通滤波：提取有效脑电频段(1-50Hz)
                    DataFilter.perform_bandpass(
                        channel_data,
                        sample_rate,
                        1.0,   # 低频截止1Hz
                        50.0,  # 高频截止50Hz
                        4,     # 4阶巴特沃斯滤波器
                        FilterTypes.BUTTERWORTH.value,
                        0
                    )
                    
                    # 更新处理后的数据
                    display_data[:, ch] = channel_data
                    
            except Exception as e:
                self.log_message(f"滤波处理出错: {str(e)}", "error")
            
            num_samples = display_data.shape[0]

            # Shannon | 2025.5.27 | 实现动态缓冲区管理
            if not hasattr(self, 'data_buffer'):
                self.display_samples = self.sample_rate * self.display_duration
                self.current_index = 0
                self.data_buffer = np.zeros((8, self.display_samples))  # 固定8通道

            # 更新EEG显示数据缓冲区
            remaining_space = self.display_samples - self.current_index
            if num_samples <= remaining_space:
                for ch in range(8):
                    self.data_buffer[ch, self.current_index:self.current_index + num_samples] = display_data[:, ch]
                self.current_index += num_samples
            else:
                for ch in range(8):
                    self.data_buffer[ch, self.current_index:] = display_data[:remaining_space, ch]
                    self.data_buffer[ch, :num_samples - remaining_space] = display_data[remaining_space:, ch]
                self.current_index = (self.current_index + num_samples) % self.display_samples

            # Shannon | 2025.5.27 | 实现实时显示更新
            # 计算显示数据的时间轴
            start_idx = (self.current_index - self.display_samples) % self.display_samples
            indices = (np.arange(self.display_samples) + start_idx) % self.display_samples
            time_values = np.linspace(-self.display_duration, 0, self.display_samples)

            # 更新每个通道的波形显示
            for ch in range(8):
                if ch < len(self.curves):
                    y_data = self.data_buffer[ch, indices]
                    self.curves[ch].setData(time_values, y_data)
                    
                    # 计算并显示RMS值
                    if ch < len(self.uVrms_labels):
                        rms = np.sqrt(np.mean(np.square(y_data)))
                        self.uVrms_labels[ch].setText(f"{rms:.2f} uVrms")

            # Shannon | 2026.6.18 | 实现120点窗口+60点步长的滚动预测缓冲
            if self.attention_classifier is not None:
                completed_windows = self.prediction_buffer.append_samples(classification_data)
                for window_data in completed_windows:
                    processed_data = self.process_data(window_data)

                    if processed_data is not None:
                        # 执行预测并获取结果
                        prediction = self.attention_classifier.predict(processed_data)
                        probability = self.attention_classifier.predict_proba(processed_data)

                        # 更新历史数据
                        self.attention_history = np.roll(self.attention_history, -1)
                        self.attention_history[-1] = prediction[0]  # 0: 不集中, 1: 集中

                        self.probability_history = np.roll(self.probability_history, -1)
                        self.probability_history[-1] = probability[0, 1]  # 注意力集中的概率

                        self.eeg_summary.update({
                            'prediction': int(prediction[0]),
                            'probability': float(probability[0, 1]),
                            'timestamp': QDateTime.currentDateTime().toString("HH:mm:ss"),
                            'total_windows': self.eeg_summary['total_windows'] + 1
                        })

                        # 更新显示图表
                        step_time_points = np.linspace(
                            -self.attention_history_seconds,
                            0,
                            len(self.attention_history) + 1
                        )
                        self.attention_level_curve.setData(step_time_points, self.attention_history)
                        self.attention_prob_curve.setData(self.time_points, self.probability_history)
                        self.update_multimodal_summary()

        except Exception as e:
            error_msg = f"更新绘图时出错: {str(e)}"
            self.log_message(error_msg, "error")
            self.log_message(traceback.format_exc(), "error")
            # 不要在这里弹出消息框，因为可能会导致界面卡死

    def process_data(self, data, window_size=120):
        """
        Shannon | 2025.5.27
        处理EEG数据用于注意力预测，包括数据验证、标准化和格式转换
        
        参数:
            data: shape (channels, samples) 的EEG数据
            window_size: 窗口大小，默认120（对应0.24秒，采样率为500Hz）
        
        返回:
            processed_data: shape (1, channels, window_size) 的处理后数据
                          或在数据无效时返回None
        """
        try:
            # 1. 数据完整性检查
            if data.shape[1] < window_size:
                print(f"数据点不足: {data.shape[1]} < {window_size}")
                return None
                
            # 2. 截取最近的window_size个采样点
            data = data[:, -window_size:].copy()
            
            # 3. 对每个通道进行标准化
            # 标准化公式：(x - mean(x)) / (std(x) + eps)
            # eps=1e-8 防止除零错误
            for i in range(data.shape[0]):
                data[i] = (data[i] - np.mean(data[i])) / (np.std(data[i]) + 1e-8)
            
            # 4. 重塑数据为模型输入格式 (1, channels, window_size)
            processed_data = data.reshape(1, data.shape[0], data.shape[1])
            return processed_data
            
        except Exception as e:
            print(f"数据处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def show_error(self, msg):
        """显示错误信息"""
        error_msg = f"设备错误: {msg}"
        self.log_message(error_msg, "error")
        
        # 创建错误对话框并设置样式
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("设备错误")
        error_dialog.setText(error_msg)
        error_dialog.setStandardButtons(QMessageBox.Ok)
        
        # 设置对话框样式
        error_dialog.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
            }
            QMessageBox QLabel {
                color: #333333;
                min-width: 300px;
                min-height: 50px;
                padding: 10px;
            }
            QPushButton {
                min-width: 80px;
                min-height: 24px;
            }
        """)
        
        # 显示对话框
        error_dialog.exec_()
        
        if self.eeg_thread:
            self.eeg_thread.running = False
            self.action_EEG.setText("开始读取脑电信号")
            self.log_message("EEG线程已停止", "info")

    def create_message_box(self, title, text, icon=QMessageBox.Information):
        """创建统一样式的消息框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setIcon(icon)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
            }
            QMessageBox QLabel {
                color: #333333;
                min-width: 300px;
                min-height: 50px;
                padding: 10px;
            }
            QPushButton {
                min-width: 80px;
                min-height: 24px;
            }
        """)
        return msg_box

    def new_savefile(self, timestamp):
        # 使用相对路径 ./Recordings
        path = "./Recordings"
        if not os.path.exists(path):
            os.makedirs(path)
        self.file_save_name = os.path.join(path, timestamp)
        init_data = np.zeros((self.num_channels, 1))
        DataFilter.write_file(init_data, self.file_save_name, 'w')

    def save_file(self, buffer):
        DataFilter.write_file(buffer, self.file_save_name, 'a')

    def analyze_eeg(self):
        QMessageBox.information(self, "分析中", "正在分析脑电信号，请稍候...")

    def run_mri_model(self, model_type):
        report = self.build_multimodal_report(model_type)
        self.analysis_reports.append({
            'type': model_type,
            'report': report,
            'created_at': QDateTime.currentDateTime().toString("HH:mm:ss")
        })
        self.log_message(f"{model_type}联合分析摘要已生成", "info")
        self.update_multimodal_summary()
        QMessageBox.information(self, f"{model_type}联合分析", report)

    def log_message(self, message, level="info", source=None):
        """添加日志消息到日志显示区域
        level: "info", "warning", "error"
        source: 日志来源，例如 "OpenBCI"
        """
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        color = {
            "info": "#000000",
            "warning": "#FFA500",
            "error": "#FF0000"
        }.get(level, "#000000")
        
        # 添加来源标签
        source_tag = f'[{source}] ' if source else ''
        
        formatted_message = f'<span style="color: #666666">[{timestamp}]</span> <span style="color: {color}">{source_tag}{message}</span>'
        self.log_text.append(formatted_message)
        self.statusBar.showMessage(message, 3000)  # 在状态栏显示3秒
        
        # 确保最新的消息可见
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
        # 同时打印到控制台
        print(f"[{timestamp}] {source_tag}{message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = MedicalDataVisualization()
    window.show()
    sys.exit(app.exec())
