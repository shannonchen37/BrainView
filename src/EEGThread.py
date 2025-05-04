# EEGThread.py
import time
import platform
import numpy as np
from PySide6.QtCore import QThread, Signal
import serial
import serial.tools.list_ports
from brainflow import BrainFlowInputParams, BoardShim, BoardIds
from brainflow.board_shim import LogLevels
import logging


class EEGThread(QThread):
    """Mac系统下的OpenBCI数据采集线程
    负责与OpenBCI设备通信，采集和预处理脑电数据
    """
    
    # Shannon | 2025.5.25 | 修改：使用object类型信号，支持复杂数据结构传输
    data_updated = Signal(object)  # 发送处理后的数据
    error_occurred = Signal(str)   # 发送错误信息
    activated = Signal()           # 发送设备激活信号
    new_savefile = Signal(str)     # 发送新文件创建信号
    file_save = Signal(object)     # 发送数据保存信号
    log_message = Signal(str, str)  # 发送日志信息 (消息, 级别)

    def __init__(self, buffer_size=1000):
        super().__init__()
        self.buffer_size = buffer_size
        self.running = False
        self.board = None
        self.sample_rate = 250  # Cyton默认采样率
        self.num_channels = 8   # 我们只关注8个EEG通道
        self.is_streaming = False
        self.save_buffer = None
        self.packet_modulo = 256
        self.max_interpolation_gap = 8
        self.last_packet_num = None
        self.last_eeg_sample = None
        self.last_stats_log_time = 0
        self.stream_stats = {
            'received_samples': 0,
            'dropped_samples': 0,
            'recovered_samples': 0,
            'duplicate_samples': 0,
            'large_gaps': 0
        }
        
        # 配置日志
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('OpenBCI')
        
        # 添加自定义处理器
        class SignalHandler(logging.Handler):
            def __init__(self, signal):
                super().__init__()
                self.signal = signal
            
            def emit(self, record):
                level = record.levelname.lower()
                self.signal.emit(record.getMessage(), level)
        
        self.logger.addHandler(SignalHandler(self.log_message))

    def reset_stream_state(self):
        """重置数据包连续性检查状态。"""
        self.last_packet_num = None
        self.last_eeg_sample = None
        self.last_stats_log_time = 0
        self.stream_stats = {
            'received_samples': 0,
            'dropped_samples': 0,
            'recovered_samples': 0,
            'duplicate_samples': 0,
            'large_gaps': 0
        }

    def normalize_channel_count(self, eeg_data):
        """将不同板卡返回的EEG通道数统一到8通道。"""
        if eeg_data.shape[0] > self.num_channels:
            return eeg_data[:self.num_channels, :]
        if eeg_data.shape[0] < self.num_channels:
            padding = np.zeros((self.num_channels - eeg_data.shape[0], eeg_data.shape[1]))
            return np.vstack((eeg_data, padding))
        return eeg_data

    def get_package_numbers(self, board_data):
        """读取BrainFlow package number通道，用于检测丢包和重复包。"""
        try:
            package_channel = BoardShim.get_package_num_channel(BoardIds.CYTON_BOARD)
            if package_channel < 0 or package_channel >= board_data.shape[0]:
                return None
            return board_data[package_channel].astype(int)
        except Exception as e:
            self.logger.debug(f"无法读取package number通道: {str(e)}")
            return None

    def repair_packet_gaps(self, eeg_data, package_numbers):
        """基于package number检查连续性，并对小间隙执行线性插值补点。"""
        if package_numbers is None or eeg_data.shape[1] == 0:
            self.stream_stats['received_samples'] += eeg_data.shape[1]
            return eeg_data
        if len(package_numbers) != eeg_data.shape[1]:
            self.logger.warning("package number数量与EEG采样点数量不一致，跳过丢包修复")
            self.stream_stats['received_samples'] += eeg_data.shape[1]
            return eeg_data

        repaired_samples = []
        self.stream_stats['received_samples'] += eeg_data.shape[1]

        for sample_index in range(eeg_data.shape[1]):
            packet_num = int(package_numbers[sample_index]) % self.packet_modulo
            current_sample = eeg_data[:, sample_index]

            if self.last_packet_num is None:
                repaired_samples.append(current_sample)
                self.last_packet_num = packet_num
                self.last_eeg_sample = current_sample.copy()
                continue

            if packet_num == self.last_packet_num:
                self.stream_stats['duplicate_samples'] += 1
                continue

            expected_packet = (self.last_packet_num + 1) % self.packet_modulo
            gap = (packet_num - expected_packet) % self.packet_modulo

            if gap == 0:
                repaired_samples.append(current_sample)
            elif gap <= self.max_interpolation_gap and self.last_eeg_sample is not None:
                self.stream_stats['dropped_samples'] += gap
                self.stream_stats['recovered_samples'] += gap
                for step in range(1, gap + 1):
                    ratio = step / (gap + 1)
                    interpolated = self.last_eeg_sample + (current_sample - self.last_eeg_sample) * ratio
                    repaired_samples.append(interpolated)
                repaired_samples.append(current_sample)
                self.logger.warning(f"检测到{gap}个丢失采样点，已使用线性插值补齐")
            else:
                self.stream_stats['dropped_samples'] += gap
                self.stream_stats['large_gaps'] += 1
                repaired_samples.append(current_sample)
                self.logger.warning(f"检测到较大数据包间隙({gap}点)，跳过插值以避免引入伪迹")

            self.last_packet_num = packet_num
            self.last_eeg_sample = current_sample.copy()

        if not repaired_samples:
            return np.empty((self.num_channels, 0))
        return np.column_stack(repaired_samples)

    def maybe_log_stream_stats(self, current_time):
        """定期输出数据流质量统计，避免日志刷屏。"""
        if current_time - self.last_stats_log_time < 10:
            return
        self.last_stats_log_time = current_time
        stats = self.stream_stats
        self.logger.info(
            "数据流统计: "
            f"接收{stats['received_samples']}点, "
            f"插值补点{stats['recovered_samples']}点, "
            f"重复包{stats['duplicate_samples']}点, "
            f"大间隙{stats['large_gaps']}次"
        )
        
    def setup_board(self):
        """设置OpenBCI板"""
        try:
            if self.board:
                try:
                    if self.is_streaming:
                        self.board.stop_stream()
                    self.board.release_session()
                except:
                    pass
                self.board = None
                self.is_streaming = False
            
            self.logger.info("开始设置OpenBCI板...")
            ports = list(serial.tools.list_ports.comports())
            self.logger.debug(f"发现串口设备: {ports}")
            
            port = None
            for p in ports:
                if 'usbserial' in p.device.lower():
                    port = p.device
                    self.logger.info(f"找到OpenBCI设备: {p.device}")
                    break
                    
            if not port:
                raise Exception("未找到OpenBCI设备")
                
            # 初始化设备
            try:
                self.logger.info("尝试串口通信...")
                with serial.Serial(port, 115200, timeout=3) as ser:
                    time.sleep(3)
                    ser.write(b's')  # 停止数据流
                    time.sleep(1)
                    ser.reset_input_buffer()
                    
                    # 发送版本查询命令
                    ser.write(b'v')
                    time.sleep(0.5)
                    # 软重置
                    ser.write(b'r')
                    time.sleep(2)
                    ser.reset_input_buffer()
                    
                    # 默认通道设置命令
                    ser.write(b'd')
                    time.sleep(1)
                    
                    # 配置所有通道
                    for i in range(1, 9):
                        cmd = f"x{i}0110X".encode()  # 增益60，连接到正常输入
                        ser.write(cmd)
                        time.sleep(0.1)
                    time.sleep(1)
                    
                    # 设置采样率250Hz
                    ser.write(b'~6')
                    time.sleep(1)
                    # 停止数据流
                    ser.write(b's')
                    time.sleep(1)
                
                self.logger.info("串口配置完成")
                
            except Exception as e:
                self.logger.error(f"串口通信失败: {str(e)}")
                raise
            
            # 设置BrainFlow参数
            params = BrainFlowInputParams()
            params.serial_port = port
            params.timeout = 15
            params.serial_number = ''
            
            self.board = BoardShim(BoardIds.CYTON_BOARD, params)
            BoardShim.enable_dev_board_logger()
            BoardShim.set_log_level(LogLevels.LEVEL_DEBUG.value)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"第{attempt+1}次尝试初始化...")
                    time.sleep(4)
                    self.board.prepare_session()
                    time.sleep(2)
                    self.activated.emit()
                    self.logger.info("设备初始化成功！")
                    self.running = True
                    return True
                    
                except Exception as e:
                    self.logger.error(f"初始化尝试 {attempt+1} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        try:
                            with serial.Serial(port, 115200, timeout=3) as ser:
                                ser.write(b's')
                                time.sleep(1)
                                ser.write(b'r')
                                time.sleep(3)
                                ser.write(b's')
                                time.sleep(1)
                        except:
                            pass
                        continue
                    else:
                        raise e
                    
        except Exception as e:
            self.logger.error(f"设备初始化失败: {str(e)}")
            self.running = False
            self.is_streaming = False
            return False
            
    def run(self):
        """线程主循环
        负责数据采集、预处理和发送
        """
        try:
            if not self.setup_board():
                return
                
            self.board.start_stream()
            self.is_streaming = True
            self.reset_stream_state()
            self.logger.info("数据流已启动")
            
            # Shannon | 2025.5.25 | 优化：获取设备参数
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)
            sample_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD)
            
            self.logger.info(f"EEG通道索引: {eeg_channels}")
            self.logger.info(f"采样率: {sample_rate} Hz")
            
            chunk_size = max(int(sample_rate * 0.1), 1)  # 每次读取0.1秒的数据
            
            # 初始化保存缓冲区
            self.save_buffer = np.empty((len(eeg_channels), 0))
            timestamp = time.time()
            local_time = time.localtime(timestamp)
            formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
            self.new_savefile.emit('Brainflow-RAW'+formatted_time+'.txt')
            
            last_data_time = time.time()
            no_data_count = 0
            rec_num = 0
            
            # Shannon | 2025.5.25 | 优化：数据归一化参数
            scale_factor = 1000.0  # 将微伏转换为毫伏
            
            while self.running:
                try:
                    current_time = time.time()
                    # 使用消费式读取，避免重复处理BrainFlow环形缓存中的重叠样本
                    data = self.board.get_board_data(chunk_size)
                    
                    if data.size > 0:
                        # Shannon | 2025.5.25 | 优化：数据处理流程
                        if len(eeg_channels) > 0:
                            package_numbers = self.get_package_numbers(data)
                            # 提取EEG数据
                            eeg_data = data[eeg_channels]
                            
                            # 记录原始数据信息
                            self.logger.debug(f"原始EEG数据形状: {eeg_data.shape}")
                            
                            # 保存原始数据
                            self.save_buffer = np.concatenate((self.save_buffer, eeg_data), axis=1)
                            rec_num += 1
                            
                            # Shannon | 2025.5.25 | 优化：确保8通道数据
                            eeg_data = self.normalize_channel_count(eeg_data)

                            # Shannon | 2026.6.18 | 新增：检查package number，修复小间隙丢包
                            eeg_data = self.repair_packet_gaps(eeg_data, package_numbers)
                            if eeg_data.shape[1] == 0:
                                continue
                            
                            # Shannon | 2025.5.25 | 优化：数据预处理
                            # 数据归一化
                            eeg_data = eeg_data / scale_factor  # 转换为毫伏
                            
                            # 数据中心化（去除直流分量）
                            eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
                            
                            # 转置数据为 (samples, channels) 格式
                            eeg_data = eeg_data.T
                            
                            # 记录处理后的数据信息
                            self.logger.debug(f"处理后数据形状: {eeg_data.shape}")
                            self.logger.debug(f"处理后数据范围: [{np.min(eeg_data):.2f}mV, {np.max(eeg_data):.2f}mV]")
                            
                            # Shannon | 2025.5.25 | 优化：使用字典传递完整信息
                            data_dict = {
                                'data': eeg_data,                    # (samples, channels) 格式的数据
                                'channels': list(range(8)),          # 通道列表
                                'sample_rate': sample_rate,          # 采样率
                                'timestamp': current_time,           # 时间戳
                                'chunk_size': chunk_size,            # 数据块大小
                                'stream_stats': self.stream_stats.copy()
                            }
                            
                            # 发送数据
                            self.data_updated.emit(data_dict)
                            self.maybe_log_stream_stats(current_time)
                        
                            # 定期保存数据
                            if rec_num % 33 == 0:
                                rec_num = 0
                                if self.save_buffer.size > 0:
                                    self.file_save.emit(self.save_buffer)
                                self.save_buffer = np.empty((len(eeg_channels), 0))
                        
                        last_data_time = current_time
                        no_data_count = 0
                    else:
                        no_data_count += 1
                        if current_time - last_data_time > 3:
                            self.logger.warning("超过3秒没有接收到数据")
                            if no_data_count > 30:
                                self.logger.error("设备可能已断开连接，尝试重新初始化...")
                                if self.setup_board():
                                    self.board.start_stream()
                                    self.is_streaming = True
                                    self.reset_stream_state()
                                    no_data_count = 0
                                    continue
                                else:
                                    break
                    
                    # 动态调整休眠时间
                    sleep_time = 0.05 if data.size > chunk_size * 1.5 else 0.1
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"数据获取错误: {str(e)}")
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"数据采集错误: {str(e)}")
            self.error_occurred.emit(str(e))
            
        finally:
            self.cleanup()
                
    def cleanup(self):
        """清理资源"""
        try:
            if self.board:
                if self.is_streaming:
                    self.board.stop_stream()
                    self.is_streaming = False
                self.board.release_session()
                self.board = None
            self.running = False
        except Exception as e:
            self.logger.error(f"清理资源错误: {str(e)}")

    def stop(self):
        """停止数据采集"""
        self.logger.info("正在停止数据采集...")
        self.running = False
        self.cleanup()
