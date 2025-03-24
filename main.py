import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                          QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
                          QStackedWidget, QFrame, QFileDialog, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QMutex
from PyQt6.QtGui import QPixmap, QImage, QFont
import torch.serialization
from numpy._core.multiarray import scalar
import numpy

# 设置PyTorch环境变量以避免内存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 添加需要的安全全局变量到允许列表
torch.serialization.add_safe_globals([
    scalar,          # numpy标量
    numpy.dtype,     # numpy数据类型
    np.dtype,        # numpy数据类型别名
    np.ndarray,      # numpy数组
    type(None),      # NoneType
    slice,           # 切片类型
    type             # type类型自身
])

# 全局样式
BUTTON_STYLE = """
QPushButton {
    background-color: #3498db;
    color: white;
    border-radius: 5px;
    font-size: 16px;
    min-height: 40px;
    padding: 10px;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:pressed {
    background-color: #1c6ea4;
}
"""

TITLE_STYLE = """
QLabel {
    font-size: 24px;
    font-weight: bold;
    color: #2c3e50;
    padding: 10px;
}
"""

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI医疗超声诊断辅助软件系统")
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # 设置窗口大小为屏幕的60%
        window_width = int(screen_width * 0.6)
        window_height = int(screen_height * 0.6)
        self.resize(window_width, window_height)
        
        # 居中显示窗口
        self.move(int((screen_width - window_width) / 2),
                 int((screen_height - window_height) / 2))
        
        # 设置窗口为可调整大小
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建垂直布局
        layout = QVBoxLayout(central_widget)
        
        # 创建堆叠部件，用于页面切换
        self.stack = QStackedWidget()
        
        # 创建主页
        self.main_page = MainPage(self)
        self.stack.addWidget(self.main_page)
        
        # 创建鼠旁路阻滞（大鼠）模块页
        self.rat_block_page = RatBlockPage(self)
        self.stack.addWidget(self.rat_block_page)
        
        # 创建胃容积计算模块页
        self.gastric_volume_page = GastricVolumePage(self)
        self.stack.addWidget(self.gastric_volume_page)
        
        layout.addWidget(self.stack)
        
    def show_main_page(self):
        self.stack.setCurrentIndex(0)
        
    def show_rat_block_page(self):
        self.stack.setCurrentIndex(1)
        
    def show_gastric_volume_page(self):
        self.stack.setCurrentIndex(2)

# 主界面类
class MainPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("AI医疗超声诊断辅助软件系统")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(TITLE_STYLE)
        layout.addWidget(title_label)
        
        # 模块按钮网格
        grid_layout = QGridLayout()
        
        # 分割类模块按钮
        seg_title = QLabel("分割类模块")
        seg_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        seg_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        grid_layout.addWidget(seg_title, 0, 0, 1, 3)
        
        # 椎旁阻滞（大鼠）按钮
        rat_btn = QPushButton("椎旁阻滞（大鼠）")
        rat_btn.setStyleSheet(BUTTON_STYLE)
        rat_btn.clicked.connect(self.main_window.show_rat_block_page)
        grid_layout.addWidget(rat_btn, 1, 0)
        
        # 椎旁阻滞（人）按钮 - 暂不实现
        human_btn = QPushButton("椎旁阻滞（人）")
        human_btn.setStyleSheet(BUTTON_STYLE)
        human_btn.setEnabled(False)
        grid_layout.addWidget(human_btn, 1, 1)
        
        # 臂丛神经阻滞按钮 - 暂不实现
        arm_btn = QPushButton("臂丛神经阻滞")
        arm_btn.setStyleSheet(BUTTON_STYLE)
        arm_btn.setEnabled(False)
        grid_layout.addWidget(arm_btn, 1, 2)
        
        # 计算类模块按钮
        calc_title = QLabel("计算类模块")
        calc_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        calc_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        grid_layout.addWidget(calc_title, 2, 0, 1, 3)
        
        # 胃内容积计算按钮
        gastric_btn = QPushButton("胃内容积计算")
        gastric_btn.setStyleSheet(BUTTON_STYLE)
        gastric_btn.clicked.connect(self.main_window.show_gastric_volume_page)
        grid_layout.addWidget(gastric_btn, 3, 0)
        
        # 下腔静脉变异计算按钮 - 暂不实现
        vein_btn = QPushButton("下腔静脉变异计算")
        vein_btn.setStyleSheet(BUTTON_STYLE)
        vein_btn.setEnabled(False)
        grid_layout.addWidget(vein_btn, 3, 1)
        
        # 胸骨旁长轴心脏计算按钮 - 暂不实现
        heart_btn = QPushButton("胸骨旁长轴心脏计算")
        heart_btn.setStyleSheet(BUTTON_STYLE)
        heart_btn.setEnabled(False)
        grid_layout.addWidget(heart_btn, 3, 2)
        
        layout.addLayout(grid_layout)
        
        # 退出按钮
        exit_btn = QPushButton("退出")
        exit_btn.setStyleSheet(BUTTON_STYLE)
        exit_btn.clicked.connect(QApplication.quit)
        
        # 版本信息
        version_label = QLabel("版本: v1.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # 底部栏
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(exit_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(version_label)
        
        layout.addStretch()
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)

# 模型推理的基础类
class ModelInference:
    def __init__(self, model_path, model_name, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, model_name, num_classes)
        
    def load_model(self, model_path, model_name, num_classes):
        import segmentation_models_pytorch as smp
        
        # 根据模型名称创建模型，使用与训练时完全相同的配置
        models = {
            'Unet': smp.Unet,
            'UnetPlusPlus': smp.UnetPlusPlus,
            'MAnet': smp.MAnet,
            'PSPNet': smp.PSPNet
        }
        
        model = models[model_name](
            encoder_name="resnet50",
            encoder_weights="imagenet",  # 与训练时保持一致
            in_channels=3,
            classes=num_classes,
        )
        
        # 加载模型权重，添加错误处理
        try:
            # 尝试方法1：使用weights_only=False
            print(f"尝试加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载成功!")
        except Exception as e:
            print(f"加载失败，错误: {str(e)}")
            print("尝试备选加载方法...")
            
            try:
                # 配置临时环境变量
                os.environ["PYTORCH_ENABLE_UNSAFE_LOAD"] = "1"
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("使用UNSAFE_LOAD成功加载模型!")
            except Exception as e2:
                print(f"备选方法也失败: {str(e2)}")
                print("尝试最终方法...")
                
                # 作为最后的选择，尝试简单加载不检查键
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    else:
                        # 直接加载权重
                        model.load_state_dict(checkpoint, strict=False)
                    print("使用非严格模式成功加载模型!")
                except Exception as e3:
                    print(f"所有加载方法均失败: {str(e3)}")
                    print("将使用未初始化的模型，预测结果可能不正确")
        
        model = model.to(self.device)
        model.eval()
        
        # 进行一次测试推理，确保模型工作正常
        try:
            print("测试模型推理...")
            # 创建一个随机输入 (1,3,512,512)
            dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
            with torch.no_grad():
                test_output = model(dummy_input)
            print(f"测试推理成功！输出形状: {test_output.shape}")
        except Exception as e:
            print(f"测试推理失败: {str(e)}")
            print("模型可能加载不完全或者结构不兼容")
        
        return model
    
    def preprocess_image(self, image):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # 预处理图像，转换为模型输入格式
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA图
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # 应用变换
        augmented = transform(image=image)
        tensor_image = augmented['image'].unsqueeze(0).to(self.device)
        
        return tensor_image
    
    def predict(self, image):
        # 预处理图像
        tensor_image = self.preprocess_image(image)
        
        # 进行推理
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()[0]
            
        # 转换预测结果为原始图像大小
        predictions = cv2.resize(predictions.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return predictions

# 视频处理线程
class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)  # 新增错误信号
    
    def __init__(self, model, video_source=0, parent=None):
        super().__init__(parent)
        self.model = model
        self.running = False
        self.camera = None
        self.frame_rate = 60  # 目标帧率
        self.video_source = video_source  # 可以是摄像头索引或视频文件路径
        self.temp_video_path = None  # 临时转换后的视频路径
        self.skip_frames = 0  # 跳帧计数器
        self.process_every_n_frames = 1  # 每N帧处理一次
        self.resize_factor = 1.0  # 处理时的缩放因子
        self._mutex = QMutex()  # 添加互斥锁以安全控制状态
        
    def convert_video(self, video_path):
        """转换视频格式为兼容格式"""
        try:
            import tempfile
            import subprocess
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            self.temp_video_path = os.path.join(temp_dir, f"converted_{os.path.basename(video_path)}")
            
            # 使用ffmpeg转换视频
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',  # 使用H.264编码
                '-preset', 'ultrafast',  # 编码速度, 从medium改为ultrafast
                '-tune', 'zerolatency',  # 优化低延迟
                '-crf', '28',  # 视频质量, 从23提高到28(值越大质量越低但处理更快)
                '-r', '60',  # 强制60fps输出
                '-y',  # 覆盖已存在的文件
                self.temp_video_path
            ]
            
            # 执行转换 - 使用二进制模式处理输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False  # 使用二进制模式而不是文本模式
            )
            
            # 等待转换完成
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                # 安全地处理二进制错误输出
                error_msg = stderr.decode('utf-8', errors='replace')
                raise Exception(f"视频转换失败: {error_msg[:200]}...")
            
            return self.temp_video_path
            
        except Exception as e:
            raise Exception(f"视频转换出错: {str(e)}")
    
    def stop(self):
        """安全停止线程"""
        self._mutex.lock()
        self.running = False
        self._mutex.unlock()
        
        # 释放摄像头资源
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
            
        # 删除临时文件
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
                self.temp_video_path = None
            except:
                pass
    
    def run(self):
        """线程主函数"""
        self._mutex.lock()
        self.running = True
        self._mutex.unlock()
        
        try:
            # 如果是视频文件，先进行格式转换
            if isinstance(self.video_source, str):
                try:
                    self.video_source = self.convert_video(self.video_source)
                except Exception as e:
                    self.error_occurred.emit(f"视频转换失败: {str(e)}")
                    return
            
            # 打开摄像头或视频文件
            self.camera = cv2.VideoCapture(self.video_source)
            if not self.camera.isOpened():
                raise ValueError("无法打开视频源")
            
            # 设置分辨率和缓冲区大小
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置OpenCV缓冲区大小
            
            # 控制帧率
            frame_time = 1.0 / self.frame_rate
            frame_count = 0
            
            # 性能统计
            start_time_total = time.time()
            frames_processed = 0
            
            # 准备上一帧的预测结果（用于跳帧时使用）
            last_prediction = None
            
            while self.running:
                loop_start_time = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    # 如果是视频文件，到达末尾时重新开始
                    if isinstance(self.video_source, str):
                        self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                frame_count += 1
                
                # 跳帧处理: 只处理每n帧中的一帧
                if frame_count % self.process_every_n_frames == 0:
                    # 可选：调整分辨率进行处理（如果需要提高性能）
                    if self.resize_factor != 1.0:
                        process_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                    else:
                        process_frame = frame
                    
                    # 模型推理
                    prediction = self.model.predict(process_frame)
                    last_prediction = prediction
                    
                    # 如果是缩小的帧，需要将预测结果放大回原始尺寸
                    if self.resize_factor != 1.0:
                        prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
                    
                    # 发送原始帧和预测结果
                    self.update_frame.emit(frame, prediction)
                    frames_processed += 1
                elif last_prediction is not None:
                    # 使用上一帧的预测结果，实现跳帧但不影响显示效果
                    # 如果预测结果与当前帧大小不匹配，进行调整
                    if last_prediction.shape[:2] != frame.shape[:2]:
                        resized_prediction = cv2.resize(last_prediction, (frame.shape[1], frame.shape[0]), 
                                                    interpolation=cv2.INTER_NEAREST)
                        self.update_frame.emit(frame, resized_prediction)
                    else:
                        self.update_frame.emit(frame, last_prediction)
                    frames_processed += 1
                
                # 控制帧率
                processing_time = time.time() - loop_start_time
                if processing_time < frame_time:
                    time.sleep(frame_time - processing_time)
                
                # 自适应调整处理频率
                elapsed = time.time() - start_time_total
                if elapsed > 3.0:  # 每3秒评估一次性能
                    actual_fps = frames_processed / elapsed
                    # 根据实际帧率动态调整处理策略
                    if actual_fps < 30:  # 如果帧率太低
                        self.process_every_n_frames = min(4, self.process_every_n_frames + 1)  # 增加跳帧
                        if self.resize_factor > 0.5:
                            self.resize_factor -= 0.1  # 逐渐降低处理分辨率
                    elif actual_fps > 55:  # 如果帧率很好
                        if self.process_every_n_frames > 1:
                            self.process_every_n_frames -= 1  # 减少跳帧
                        if self.resize_factor < 1.0:
                            self.resize_factor += 0.1  # 逐渐提高处理分辨率
                            
                    # 重置计数器
                    start_time_total = time.time()
                    frames_processed = 0
                    
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # 释放资源
            self.stop()

# 椎旁阻滞（大鼠）页面
class RatBlockPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model_path = "/root/sj-tmp/save_models_new/rat/Unet/best_model.pth"
        
        # 检查模型路径
        if not os.path.exists(self.model_path):
            print(f"警告：模型文件不存在: {self.model_path}")
            print("尝试搜索替代模型文件...")
            
            # 检查其他可能的模型路径
            possible_paths = [
                "/root/sj-tmp/save_models_new/rat/Unet/best_model.pth",
                "/root/sj-tmp/save_models/rat/UnetPlusPlus/best_model.pth",
                "/root/sj-tmp/save_models/rat/Unet/best_model.pth"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"找到替代模型: {path}")
                    self.model_path = path
                    # 根据模型路径判断模型类型
                    if "UnetPlusPlus" in path:
                        self.model_type = "UnetPlusPlus"
                    elif "Unet" in path:
                        self.model_type = "Unet"
                    else:
                        self.model_type = "UnetPlusPlus"  # 默认值
                    break
            else:
                print("未找到可用模型，请确保模型文件存在")
        else:
            self.model_type = "UnetPlusPlus"  # 默认使用UnetPlusPlus
        
        # 使用5类（背景+4个类别：椎间孔IF、关节突AP、棘突SP、髂骨IB）
        self.model = ModelInference(self.model_path, self.model_type, 5)
        
        self.video_thread = None
        self.recording = False
        self.current_file = None  # 当前处理的文件路径
        self.max_file_size = 500 * 1024 * 1024  # 最大文件大小（500MB）
        
        # 识别相关参数
        self.frame_timestamps = []  # 帧时间戳列表
        self.frames_with_nerve = []  # 存储包含神经区域的帧
        self.first_ultrasound_time = None  # 首次出现超声图像的时间
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题栏
        title_layout = QHBoxLayout()
        title_label = QLabel("椎旁阻滞（大鼠）模块")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        back_btn = QPushButton("返回主界面")
        back_btn.clicked.connect(self.main_window.show_main_page)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(back_btn)
        layout.addLayout(title_layout)
        
        # 主内容区
        content_layout = QHBoxLayout()
        
        # 左侧 - 探头示意图和操作说明
        left_layout = QVBoxLayout()
        
        # 探头示意图框架
        probe_frame = QFrame()
        probe_frame.setFrameShape(QFrame.Shape.StyledPanel)
        probe_layout = QVBoxLayout()
        probe_title = QLabel("操作探头示意图")
        probe_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        probe_title.setStyleSheet("font-weight: bold;")
        probe_layout.addWidget(probe_title)
        
        # 探头示意图
        probe_image = QLabel()
        probe_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        probe_image.setMinimumSize(200, 150)
        probe_image.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        
        # 加载示意图（如果有的话）
        try:
            pixmap = QPixmap("resources/probe_rat_illustration.png")
            probe_image.setPixmap(pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio))
        except:
            probe_image.setText("探头示意图位置")
            
        probe_layout.addWidget(probe_image)
        probe_frame.setLayout(probe_layout)
        left_layout.addWidget(probe_frame)
        
        # 操作说明
        instructions_frame = QFrame()
        instructions_frame.setFrameShape(QFrame.Shape.StyledPanel)
        instructions_layout = QVBoxLayout()
        instructions_title = QLabel("操作说明")
        instructions_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions_title.setStyleSheet("font-weight: bold;")
        instructions_layout.addWidget(instructions_title)
        instructions_text = QLabel(
            "1. 将超声探头置于大鼠椎旁部位\n"
            "2. 点击开始识别按钮，开始实时识别神经结构\n"
            "3. 系统将自动识别并标记神经结构\n"
            "4. 点击停止识别按钮结束识别过程"
        )
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        instructions_frame.setLayout(instructions_layout)
        left_layout.addWidget(instructions_frame)
        
        # 颜色图例
        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.Shape.StyledPanel)
        legend_layout = QVBoxLayout()
        legend_title = QLabel("颜色图例")
        legend_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend_title.setStyleSheet("font-weight: bold;")
        legend_layout.addWidget(legend_title)
        
        # 椎间孔(IF) - 蓝色
        legend_item1_layout = QHBoxLayout()
        blue_box = QLabel()
        blue_box.setFixedSize(20, 20)
        blue_box.setStyleSheet("background-color: #0000ff;")
        legend_item1_layout.addWidget(blue_box)
        legend_item1_layout.addWidget(QLabel("椎间孔(IF)"))
        legend_layout.addLayout(legend_item1_layout)
        
        # 关节突(AP) - 绿色
        legend_item2_layout = QHBoxLayout()
        green_box = QLabel()
        green_box.setFixedSize(20, 20)
        green_box.setStyleSheet("background-color: #00ff00;")
        legend_item2_layout.addWidget(green_box)
        legend_item2_layout.addWidget(QLabel("关节突(AP)"))
        legend_layout.addLayout(legend_item2_layout)
        
        # 棘突(SP) - 红色
        legend_item3_layout = QHBoxLayout()
        red_box = QLabel()
        red_box.setFixedSize(20, 20)
        red_box.setStyleSheet("background-color: #ff0000;")
        legend_item3_layout.addWidget(red_box)
        legend_item3_layout.addWidget(QLabel("棘突(SP)"))
        legend_layout.addLayout(legend_item3_layout)
        
        # 髂骨(IB) - 黄色
        legend_item4_layout = QHBoxLayout()
        yellow_box = QLabel()
        yellow_box.setFixedSize(20, 20)
        yellow_box.setStyleSheet("background-color: #ffff00;")
        legend_item4_layout.addWidget(yellow_box)
        legend_item4_layout.addWidget(QLabel("髂骨(IB)"))
        legend_layout.addLayout(legend_item4_layout)
        
        legend_frame.setLayout(legend_layout)
        left_layout.addWidget(legend_frame)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("开始识别")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.screenshot_btn = QPushButton("截图")
        self.screenshot_btn.clicked.connect(self.capture_screenshot)
        
        self.upload_btn = QPushButton("上传文件")
        self.upload_btn.clicked.connect(self.upload_file)
        
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.screenshot_btn)
        control_layout.addWidget(self.upload_btn)
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        # 右侧布局 - 图像显示
        right_layout = QVBoxLayout()
        
        # 图像显示区
        image_layout = QHBoxLayout()
        
        # 原始超声图像显示
        original_container = QVBoxLayout()
        original_title = QLabel("原始超声图像")
        original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image = QLabel()
        self.original_image.setMinimumSize(400, 300)
        self.original_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image.setStyleSheet("border: 1px solid #cccccc;")
        self.original_image.setText("等待图像...")
        original_container.addWidget(original_title)
        original_container.addWidget(self.original_image)
        image_layout.addLayout(original_container)
        
        # 分割结果显示
        segmented_container = QVBoxLayout()
        segmented_title = QLabel("识别结果")
        segmented_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmented_image = QLabel()
        self.segmented_image.setMinimumSize(400, 300)
        self.segmented_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmented_image.setStyleSheet("border: 1px solid #cccccc;")
        self.segmented_image.setText("等待识别结果...")
        segmented_container.addWidget(segmented_title)
        segmented_container.addWidget(self.segmented_image)
        image_layout.addLayout(segmented_container)
        
        right_layout.addLayout(image_layout)
        
        # 状态显示区
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_layout = QVBoxLayout()
        status_title = QLabel("识别状态")
        status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_title.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(status_title)
        
        # 添加结果标签
        self.result_label = QLabel("识别结果将在这里显示")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 12pt; margin: 10px;")
        status_layout.addWidget(self.result_label)
        
        self.status_label = QLabel("等待开始识别...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14pt; margin: 10px;")
        status_layout.addWidget(self.status_label)
        
        status_frame.setLayout(status_layout)
        right_layout.addWidget(status_frame)
        
        # 组合左右布局
        content_layout.addLayout(left_layout, 1)  # 1份空间
        content_layout.addLayout(right_layout, 2)  # 2份空间
        
        layout.addLayout(content_layout)
        
        self.setLayout(layout)

    def toggle_recording(self):
        if not self.recording:
            # 开始记录
            self.record_btn.setText("停止识别")
            self.recording = True
            
            # 重置采集状态和数据
            self.frames_with_nerve = []
            self.frame_timestamps = []
            self.first_ultrasound_time = None
            
            self.status_label.setText("识别中...")
            
            # 启动视频线程
            if self.video_thread is None or not self.video_thread.running:
                self.video_thread = VideoThread(self.model)
                self.video_thread.update_frame.connect(self.update_frame)
                self.video_thread.error_occurred.connect(self.handle_video_error)
                self.video_thread.start()
        else:
            # 停止识别
            self.record_btn.setText("开始识别")
            self.recording = False
            self.status_label.setText(f"识别已停止 - 共处理 {len(self.frame_timestamps)} 帧")
                
    def update_frame(self, frame, prediction):
        current_time = time.time()
        self.frame_timestamps.append(current_time)
        
        # 如果这是第一帧，记录为超声图像首次出现时间
        if self.first_ultrasound_time is None:
            self.first_ultrasound_time = current_time
        
        # 显示原始帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_image.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.original_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # 显示分割结果
        # 创建彩色掩码
        color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        
        # 安全地为每个类别创建掩码，先检查类别是否存在
        if np.any(prediction == 1):  # 类别 1 (椎间孔IF) - 蓝色
            color_mask[prediction == 1] = [0, 0, 255]  # 蓝色
        if np.any(prediction == 2):  # 类别 2 (关节突AP) - 绿色
            color_mask[prediction == 2] = [0, 255, 0]  # 绿色
        if np.any(prediction == 3):  # 类别 3 (棘突SP) - 红色
            color_mask[prediction == 3] = [255, 0, 0]  # 红色
        if np.any(prediction == 4):  # 类别 4 (髂骨IB) - 黄色
            color_mask[prediction == 4] = [255, 255, 0]  # 黄色
        
        # 将掩码与原始图像混合
        alpha = 0.5
        segmented_image = cv2.addWeighted(frame_rgb, 1-alpha, color_mask, alpha, 0)
        
        # 转换为Qt图像并显示
        h, w, ch = segmented_image.shape
        bytes_per_line = ch * w
        qt_seg_image = QImage(segmented_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.segmented_image.setPixmap(QPixmap.fromImage(qt_seg_image).scaled(
            self.segmented_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # 如果正在记录，处理神经检测逻辑
        if self.recording:
            # 检查是否检测到各种结构
            has_if = np.any(prediction == 1)  # 椎间孔
            has_ap = np.any(prediction == 2)  # 关节突
            has_sp = np.any(prediction == 3)  # 棘突
            has_ib = np.any(prediction == 4)  # 髂骨
            
            status_text = "识别中...\n"
            
            if has_if:
                # 保存包含椎间孔的帧
                self.frames_with_nerve.append(frame)
                # 更新识别状态
                if_ratio = len(self.frames_with_nerve) / len(self.frame_timestamps)
                status_text += f"检测到椎间孔(IF) ({if_ratio:.2f})\n"
            else:
                status_text += "未检测到椎间孔(IF)\n"
                
            if has_ap:
                status_text += "检测到关节突(AP)\n"
                
            if has_sp:
                status_text += "检测到棘突(SP)\n"
                
            if has_ib:
                status_text += "检测到髂骨(IB)"
            
            self.status_label.setText(status_text)
    
    def capture_screenshot(self):
        # 保存当前分割结果截图
        if hasattr(self, 'original_image') and self.original_image.pixmap():
            filename, _ = QFileDialog.getSaveFileName(self, "保存截图", "", "PNG文件 (*.png);;JPEG文件 (*.jpg *.jpeg)")
            if filename:
                self.segmented_image.pixmap().save(filename)
                
    def report_error(self):
        # 实现错误报告功能
        pass
        
    def closeEvent(self, event):
        # 关闭线程
        if self.video_thread is not None and self.video_thread.running:
            self.video_thread.stop()
        super().closeEvent(event)
                
    def upload_file(self):
        """上传图片或视频文件"""
        try:
            # 如果正在录制，先停止
            if self.recording:
                self.toggle_recording()
            
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("媒体文件 (*.jpg *.jpeg *.png *.mp4 *.avi);;所有文件 (*.*)")
            file_dialog.setViewMode(QFileDialog.ViewMode.List)
            
            if file_dialog.exec():
                file_path = file_dialog.selectedFiles()[0]
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    raise FileNotFoundError("文件不存在")
                
                # 检查文件大小
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size:
                    raise ValueError(f"文件大小超过限制（最大{self.max_file_size/1024/1024}MB）")
                
                # 检查文件类型
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']:
                    raise ValueError("不支持的文件类型")
                
                # 检查ffmpeg是否可用
                if file_ext in ['.mp4', '.avi']:
                    try:
                        import subprocess
                        subprocess.run(['ffmpeg', '-version'], capture_output=True)
                    except FileNotFoundError:
                        raise ValueError("未找到ffmpeg，请安装ffmpeg以支持视频处理")
                
                self.current_file = file_path
                
                # 停止当前视频处理（如果有）
                if self.video_thread is not None and self.video_thread.running:
                    self.video_thread.stop()
                    self.video_thread.wait()  # 等待线程完全停止
                    self.record_btn.setText("开始识别")
                
                # 根据文件类型处理
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    self.process_image(file_path)
                else:  # 视频文件
                    self.process_video(file_path)
                    
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"文件上传失败：{str(e)}")
    
    def process_image(self, image_path):
        """处理单张图片"""
        try:
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError("无法读取图片文件")
            
            # 检查图片尺寸
            if frame.shape[0] > 4096 or frame.shape[1] > 4096:
                raise ValueError("图片尺寸过大")
            
            # 进行模型推理
            prediction = self.model.predict(frame)
            
            # 重置数据和状态
            temp_recording = self.recording
            self.recording = False  # 临时设置为False防止触发update_frame中的记录逻辑
            
            # 显示原始帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.original_image.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.original_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
            
            # 显示分割结果
            # 创建彩色掩码
            color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            
            # 安全地为每个类别创建掩码，先检查类别是否存在
            if np.any(prediction == 1):  # 类别 1 (椎间孔IF) - 蓝色
                color_mask[prediction == 1] = [0, 0, 255]  # 蓝色
            if np.any(prediction == 2):  # 类别 2 (关节突AP) - 绿色
                color_mask[prediction == 2] = [0, 255, 0]  # 绿色
            if np.any(prediction == 3):  # 类别 3 (棘突SP) - 红色
                color_mask[prediction == 3] = [255, 0, 0]  # 红色
            if np.any(prediction == 4):  # 类别 4 (髂骨IB) - 黄色
                color_mask[prediction == 4] = [255, 255, 0]  # 黄色
            
            # 将掩码与原始图像混合
            alpha = 0.5
            segmented_image = cv2.addWeighted(frame_rgb, 1-alpha, color_mask, alpha, 0)
            
            # 转换为Qt图像并显示
            h, w, ch = segmented_image.shape
            bytes_per_line = ch * w
            qt_seg_image = QImage(segmented_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.segmented_image.setPixmap(QPixmap.fromImage(qt_seg_image).scaled(
                self.segmented_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
            
            # 重置收集数据
            self.frames_with_nerve = [frame]
            self.frame_timestamps = [time.time()]

            # 计算各区域面积
            if_area = np.sum(prediction == 1)
            ap_area = np.sum(prediction == 2)
            sp_area = np.sum(prediction == 3)
            ib_area = np.sum(prediction == 4)
            
            # 更新结果显示
            result_text = f"椎间孔(IF)面积: {if_area:.2f} 像素\n"
            if ap_area > 0:
                result_text += f"关节突(AP)面积: {ap_area:.2f} 像素\n"
            if sp_area > 0:
                result_text += f"棘突(SP)面积: {sp_area:.2f} 像素\n"
            if ib_area > 0:
                result_text += f"髂骨(IB)面积: {ib_area:.2f} 像素"
            self.result_label.setText(result_text)
            
            # 恢复录制状态
            self.recording = temp_recording
            self.status_label.setText("图片已处理")
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"图片处理失败：{str(e)}")
            
    def process_video(self, video_path):
        """处理视频文件"""
        try:
            # 重置数据
            self.frames_with_nerve = []
            self.frame_timestamps = []
            self.status_label.setText("处理中...")
            
            # 创建新的视频线程
            self.video_thread = VideoThread(self.model, video_path)
            self.video_thread.update_frame.connect(self.update_frame)
            self.video_thread.error_occurred.connect(self.handle_video_error)
            
            # 自动开始记录
            self.recording = True
            
            # 启动线程
            self.video_thread.start()
            self.record_btn.setText("停止识别")
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"视频处理失败：{str(e)}")
            
    def handle_video_error(self, error_msg):
        """处理视频错误"""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "错误", f"视频处理出错：{error_msg}")
        if self.video_thread is not None:
            self.video_thread.stop()
        self.record_btn.setText("开始识别")

# 胃内容积计算页面
class GastricVolumePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model_path = "/root/sj-tmp/save_models/gastric/UnetPlusPlus/best_model.pth"
        
        # 检查模型路径
        if not os.path.exists(self.model_path):
            print(f"警告：胃容积模型文件不存在: {self.model_path}")
            print("尝试搜索替代模型文件...")
            
            # 检查其他可能的模型路径
            possible_paths = [
                "/root/sj-tmp/save_models_new/gastric/UnetPlusPlus/best_model.pth",
                "/root/sj-tmp/save_models_new/gastric/Unet/best_model.pth",
                "/root/sj-tmp/save_models/gastric/Unet/best_model.pth"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"找到替代胃容积模型: {path}")
                    self.model_path = path
                    # 根据模型路径判断模型类型
                    if "UnetPlusPlus" in path:
                        self.model_type = "UnetPlusPlus"
                    elif "Unet" in path:
                        self.model_type = "Unet"
                    else:
                        self.model_type = "UnetPlusPlus"  # 默认值
                    break
            else:
                print("未找到可用胃容积模型，请确保模型文件存在")
        else:
            self.model_type = "UnetPlusPlus"  # 默认使用UnetPlusPlus
        
        # 使用2类（背景+1个类别：胃部）
        self.model = ModelInference(self.model_path, self.model_type, 2)
        
        self.video_thread = None
        self.frames_with_gastric = []  # 存储包含胃部的帧
        self.gastric_areas = []  # 存储胃部面积
        self.recording = False  # 是否正在记录
        self.current_file = None  # 当前处理的文件路径
        self.max_file_size = 500 * 1024 * 1024  # 最大文件大小（500MB）
        
        # 胃容积计算相关参数
        self.first_gastric_time = None  # 胃部首次出现的时间
        self.last_gastric_time = None  # 胃部最后出现的时间
        self.no_gastric_start_time = None  # 胃部开始消失的时间
        self.gastric_missing_duration = 3.0  # 胃部消失持续时间阈值（秒）
        self.frame_timestamps = []  # 帧时间戳列表
        self.first_ultrasound_time = None  # 首次出现超声图像的时间
        self.early_detection_threshold = 0.5  # 早期检测阈值（秒）
        self.early_detection_warning = False  # 是否发出早期检测警告
        
        # 胃容积计算参数
        self.calibration_data = {
            # 预设的校准数据: (面积, 真实容积)
            "空腹": (10000, 20),   # 空腹状态：面积10000像素，约20ml
            "半满": (50000, 150),  # 半满状态：面积50000像素，约150ml
            "充盈": (100000, 400)  # 充盈状态：面积100000像素，约400ml
        }
        self.model_type = "非线性"  # 模型类型：线性, 非线性, 指数
        self.is_calibrated = False  # 是否已校准
        
        self.init_ui()
        
    def init_ui(self):
        # 创建布局
        layout = QVBoxLayout()
        
        # 标题栏
        title_layout = QHBoxLayout()
        title_label = QLabel("胃内容积计算模块")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        back_btn = QPushButton("返回主界面")
        back_btn.clicked.connect(self.main_window.show_main_page)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(back_btn)
        layout.addLayout(title_layout)
        
        # 主内容区
        content_layout = QHBoxLayout()
        
        # 左侧 - 探头示意图和操作说明
        left_layout = QVBoxLayout()
        
        # 探头示意图框架
        probe_frame = QFrame()
        probe_frame.setFrameShape(QFrame.Shape.StyledPanel)
        probe_layout = QVBoxLayout()
        probe_title = QLabel("操作探头示意图")
        probe_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        probe_title.setStyleSheet("font-weight: bold;")
        probe_layout.addWidget(probe_title)
        
        # 探头示意图
        probe_image = QLabel()
        probe_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        probe_image.setMinimumSize(200, 150)
        probe_image.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        
        # 加载示意图（如果有的话）
        try:
            pixmap = QPixmap("resources/probe_illustration.png")
            probe_image.setPixmap(pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio))
        except:
            probe_image.setText("探头示意图位置")
            
        probe_layout.addWidget(probe_image)
        probe_frame.setLayout(probe_layout)
        left_layout.addWidget(probe_frame)
        
        # 操作说明
        instructions_frame = QFrame()
        instructions_frame.setFrameShape(QFrame.Shape.StyledPanel)
        instructions_layout = QVBoxLayout()
        instructions_title = QLabel("操作说明")
        instructions_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions_title.setStyleSheet("font-weight: bold;")
        instructions_layout.addWidget(instructions_title)
        instructions_text = QLabel(
            "1. 将超声探头置于胃部区域\n"
            "2. 点击开始采集按钮，开始记录胃部超声图像\n"
            "3. 系统将自动识别胃部并标记，计算容积\n"
            "4. 采集结束后点击停止采集，系统将计算平均胃内容积\n"
            "5. 点击校准模型可以校准容积计算参数"
        )
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        instructions_frame.setLayout(instructions_layout)
        left_layout.addWidget(instructions_frame)
        
        # 颜色图例
        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.Shape.StyledPanel)
        legend_layout = QVBoxLayout()
        legend_title = QLabel("颜色图例")
        legend_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend_title.setStyleSheet("font-weight: bold;")
        legend_layout.addWidget(legend_title)
        
        # 椎间孔(IF) - 蓝色
        legend_item1_layout = QHBoxLayout()
        blue_box = QLabel()
        blue_box.setFixedSize(20, 20)
        blue_box.setStyleSheet("background-color: #0000ff;")
        legend_item1_layout.addWidget(blue_box)
        legend_item1_layout.addWidget(QLabel("椎间孔(IF)"))
        legend_layout.addLayout(legend_item1_layout)
        
        # 关节突(AP) - 绿色
        legend_item2_layout = QHBoxLayout()
        green_box = QLabel()
        green_box.setFixedSize(20, 20)
        green_box.setStyleSheet("background-color: #00ff00;")
        legend_item2_layout.addWidget(green_box)
        legend_item2_layout.addWidget(QLabel("关节突(AP)"))
        legend_layout.addLayout(legend_item2_layout)
        
        # 棘突(SP) - 红色
        legend_item3_layout = QHBoxLayout()
        red_box = QLabel()
        red_box.setFixedSize(20, 20)
        red_box.setStyleSheet("background-color: #ff0000;")
        legend_item3_layout.addWidget(red_box)
        legend_item3_layout.addWidget(QLabel("棘突(SP)"))
        legend_layout.addLayout(legend_item3_layout)
        
        # 髂骨(IB) - 黄色
        legend_item4_layout = QHBoxLayout()
        yellow_box = QLabel()
        yellow_box.setFixedSize(20, 20)
        yellow_box.setStyleSheet("background-color: #ffff00;")
        legend_item4_layout.addWidget(yellow_box)
        legend_item4_layout.addWidget(QLabel("髂骨(IB)"))
        legend_layout.addLayout(legend_item4_layout)
        
        legend_frame.setLayout(legend_layout)
        left_layout.addWidget(legend_frame)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("开始采集")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.screenshot_btn = QPushButton("截图")
        self.screenshot_btn.clicked.connect(self.capture_screenshot)
        
        self.upload_btn = QPushButton("上传文件")
        self.upload_btn.clicked.connect(self.upload_file)
        
        self.calibrate_btn = QPushButton("校准模型")
        self.calibrate_btn.clicked.connect(self.calibrate_model)
        
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.screenshot_btn)
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.calibrate_btn)
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        # 模型状态
        self.model_status = QLabel("模型状态: 未校准")
        self.model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.model_status)
        
        # 右侧布局 - 图像显示和结果
        right_layout = QVBoxLayout()
        
        # 图像显示区
        image_layout = QHBoxLayout()
        
        # 原始超声图像显示
        original_container = QVBoxLayout()
        original_title = QLabel("原始超声图像")
        original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image = QLabel()
        self.original_image.setMinimumSize(400, 300)
        self.original_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image.setStyleSheet("border: 1px solid #cccccc;")
        self.original_image.setText("等待图像...")
        original_container.addWidget(original_title)
        original_container.addWidget(self.original_image)
        image_layout.addLayout(original_container)
        
        # 分割结果显示
        segmented_container = QVBoxLayout()
        segmented_title = QLabel("分割结果")
        segmented_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmented_image = QLabel()
        self.segmented_image.setMinimumSize(400, 300)
        self.segmented_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmented_image.setStyleSheet("border: 1px solid #cccccc;")
        self.segmented_image.setText("等待分割结果...")
        segmented_container.addWidget(segmented_title)
        segmented_container.addWidget(self.segmented_image)
        image_layout.addLayout(segmented_container)
        
        right_layout.addLayout(image_layout)
        
        # 结果显示区
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.Shape.StyledPanel)
        result_layout = QVBoxLayout()
        result_title = QLabel("计算结果")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(result_title)
        
        self.result_label = QLabel("胃内容积: 未计算")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14pt; margin: 10px;")
        result_layout.addWidget(self.result_label)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12pt; color: #666;")
        result_layout.addWidget(self.status_label)
        
        result_frame.setLayout(result_layout)
        right_layout.addWidget(result_frame)
        
        # 组合左右布局
        content_layout.addLayout(left_layout, 1)  # 1份空间
        content_layout.addLayout(right_layout, 2)  # 2份空间
        
        layout.addLayout(content_layout)
        self.setLayout(layout)
        
        # 更新模型状态显示
        self.update_model_status()

    def update_model_status(self):
        """更新模型状态显示"""
        status = f"模型状态: {'已校准' if self.is_calibrated else '未校准'} | 模型类型: {self.model_type}"
        if self.is_calibrated:
            if self.model_type == "线性":
                status += f" | 校准点: {len(self.calibration_data)}个"
            elif self.model_type == "非线性":
                status += f" | 校准点: {len(self.calibration_data)}个" 
            else:
                status += f" | 校准点: {len(self.calibration_data)}个"
        self.model_status.setText(status)
    
    def calibrate_model(self):
        """校准胃容积计算模型"""
        dialog = QDialog(self)
        dialog.setWindowTitle("校准胃容积计算模型")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # 模型类型选择
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("模型类型:"))
        model_type_combo = QComboBox()
        model_type_combo.addItems(["线性", "非线性", "指数"])
        model_type_combo.setCurrentText(self.model_type)
        model_type_layout.addWidget(model_type_combo)
        layout.addLayout(model_type_layout)
        
        # 校准数据表格
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["状态描述", "胃部面积(像素)", "实际容积(ml)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # 填充现有数据
        table.setRowCount(len(self.calibration_data))
        for i, (desc, (area, volume)) in enumerate(self.calibration_data.items()):
            table.setItem(i, 0, QTableWidgetItem(desc))
            table.setItem(i, 1, QTableWidgetItem(str(area)))
            table.setItem(i, 2, QTableWidgetItem(str(volume)))
        
        # 添加和删除行的按钮
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("添加行")
        del_btn = QPushButton("删除行")
        
        def add_row():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(f"状态{row+1}"))
            table.setItem(row, 1, QTableWidgetItem("0"))
            table.setItem(row, 2, QTableWidgetItem("0"))
            
        def del_row():
            row = table.currentRow()
            if row >= 0:
                table.removeRow(row)
        
        add_btn.clicked.connect(add_row)
        del_btn.clicked.connect(del_row)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        
        layout.addWidget(table)
        layout.addLayout(btn_layout)
        
        # 确认和取消按钮
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # 处理结果
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存模型类型
            self.model_type = model_type_combo.currentText()
            
            # 保存校准数据
            self.calibration_data = {}
            for row in range(table.rowCount()):
                desc = table.item(row, 0).text()
                try:
                    area = float(table.item(row, 1).text())
                    volume = float(table.item(row, 2).text())
                    self.calibration_data[desc] = (area, volume)
                except (ValueError, AttributeError):
                    pass  # 忽略无效数据
            
            # 设置校准状态
            self.is_calibrated = len(self.calibration_data) >= 2
            
            # 更新状态显示
            self.update_model_status()
            
            # 如果有当前数据，重新计算
            if len(self.gastric_areas) > 0:
                # 计算有效帧的胃部面积均值
                mean_area = np.mean(self.gastric_areas)
                
                # 计算最终容积（ml）
                volume = self.calculate_gastric_volume(mean_area)
                
                # 显示结果
                self.update_result_display(volume, mean_area)
    
    def calculate_gastric_volume(self, area):
        """计算胃容积
        使用校准数据和选择的模型类型计算胃容积
        """
        # 如果没有校准，使用默认系数
        if not self.is_calibrated or len(self.calibration_data) < 2:
            # 兜底默认系数
            return area * 0.01  # 简单线性系数
        
        # 提取校准数据
        calibration_points = [(area, vol) for area, vol in self.calibration_data.values() if area > 0 and vol > 0]
        if len(calibration_points) < 2:
            return area * 0.01  # 校准点不足，使用默认系数
        
        # 按面积排序
        calibration_points.sort(key=lambda x: x[0])
        areas, volumes = zip(*calibration_points)
        
        # 根据模型类型计算
        if self.model_type == "线性":
            # 线性模型: y = ax + b
            if len(areas) == 2:
                # 两点确定一条直线
                a = (volumes[1] - volumes[0]) / (areas[1] - areas[0])
                b = volumes[0] - a * areas[0]
                return a * area + b
            else:
                # 多点线性回归
                A = np.vstack([areas, np.ones(len(areas))]).T
                a, b = np.linalg.lstsq(A, volumes, rcond=None)[0]
                return a * area + b
                
        elif self.model_type == "非线性":
            # 非线性模型: y = a * x^b
            try:
                # 转换为对数空间进行线性回归
                log_areas = np.log(np.array(areas))
                log_volumes = np.log(np.array(volumes))
                A = np.vstack([log_areas, np.ones(len(log_areas))]).T
                b, log_a = np.linalg.lstsq(A, log_volumes, rcond=None)[0]
                a = np.exp(log_a)
                return a * (area ** b)
            except:
                # 如果发生数值错误，退回到简单线性模型
                A = np.vstack([areas, np.ones(len(areas))]).T
                a, b = np.linalg.lstsq(A, volumes, rcond=None)[0]
                return a * area + b
                
        elif self.model_type == "指数":
            # 指数模型: y = a * e^(b*x)
            try:
                # 转换为对数空间进行线性回归
                log_volumes = np.log(np.array(volumes))
                A = np.vstack([areas, np.ones(len(areas))]).T
                b, log_a = np.linalg.lstsq(A, log_volumes, rcond=None)[0]
                a = np.exp(log_a)
                return a * np.exp(b * area)
            except:
                # 如果发生数值错误，退回到简单线性模型
                A = np.vstack([areas, np.ones(len(areas))]).T
                a, b = np.linalg.lstsq(A, volumes, rcond=None)[0]
                return a * area + b
        
        # 默认情况：使用最接近的两点进行线性插值
        for i in range(len(areas) - 1):
            if areas[i] <= area <= areas[i+1]:
                ratio = (area - areas[i]) / (areas[i+1] - areas[i])
                return volumes[i] + ratio * (volumes[i+1] - volumes[i])
            
        # 超出范围：使用边界点的线性模型进行外推
        if area < areas[0]:
            ratio = area / areas[0]
            return volumes[0] * ratio
        else:
            ratio = (area - areas[-2]) / (areas[-1] - areas[-2])
            return volumes[-2] + ratio * (volumes[-1] - volumes[-2])
    
    def update_result_display(self, volume, area=None, frame_ratio=None):
        """更新结果显示"""
        result_text = f"胃内容积: {volume:.2f} ml\n"
        
        if area is not None:
            result_text += f"平均面积: {area:.2f} 像素\n"
        
        if len(self.frame_timestamps) > 0 and frame_ratio is None:
            frame_ratio = len(self.gastric_areas) / len(self.frame_timestamps)
            result_text += f"采集帧数: {len(self.frame_timestamps)}\n"
            result_text += f"有效帧数: {len(self.gastric_areas)}\n"
            result_text += f"帧占比: {frame_ratio:.2f}\n"
        elif frame_ratio is not None:
            result_text += f"帧占比: {frame_ratio:.2f}\n"
        
        # 添加模型信息
        if self.is_calibrated:
            result_text += f"计算模型: {self.model_type}"
        
        self.result_label.setText(result_text)
        
    def toggle_recording(self):
        if not self.recording:
            # 开始记录
            self.record_btn.setText("停止采集")
            self.recording = True
            
            # 重置采集状态和数据
            self.frames_with_gastric = []
            self.gastric_areas = []
            self.frame_timestamps = []
            self.first_gastric_time = None
            self.last_gastric_time = None
            self.no_gastric_start_time = None
            self.first_ultrasound_time = None
            self.early_detection_warning = False
            
            self.result_label.setText("采集中...")
            
            # 启动视频线程
            if self.video_thread is None or not self.video_thread.running:
                self.video_thread = VideoThread(self.model)
                self.video_thread.update_frame.connect(self.update_frame)
                self.video_thread.error_occurred.connect(self.handle_video_error)
                self.video_thread.start()
        else:
            # 停止记录并计算结果
            self.record_btn.setText("开始采集")
            self.recording = False
            
            # 计算胃内容积
            if len(self.gastric_areas) > 0:
                if self.early_detection_warning:
                    self.result_label.setText("警告：检测到早期胃部标记，请重新扫查")
                    return
                
                # 计算有效帧的胃部面积均值
                mean_area = np.mean(self.gastric_areas)
                
                # 计算最终容积（ml）
                volume = self.calculate_gastric_volume(mean_area)
                
                # 显示结果
                self.update_result_display(volume, mean_area)
            else:
                self.result_label.setText("未检测到胃部")
                
    def update_frame(self, frame, prediction):
        current_time = time.time()
        self.frame_timestamps.append(current_time)
        
        # 如果这是第一帧，记录为超声图像首次出现时间
        if self.first_ultrasound_time is None:
            self.first_ultrasound_time = current_time
        
        # 显示原始帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_image.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.original_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # 显示分割结果
        # 创建彩色掩码
        color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        
        # 安全地为胃部类别创建掩码
        gastric_mask = (prediction == 1)
        if np.any(gastric_mask):  # 类别 1 (胃部) - 红色
            color_mask[gastric_mask] = [255, 0, 0]  # 红色
        
        # 将掩码与原始图像混合
        alpha = 0.5
        segmented_image = cv2.addWeighted(frame_rgb, 1-alpha, color_mask, alpha, 0)
        
        # 转换为Qt图像并显示
        h, w, ch = segmented_image.shape
        bytes_per_line = ch * w
        qt_seg_image = QImage(segmented_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.segmented_image.setPixmap(QPixmap.fromImage(qt_seg_image).scaled(
            self.segmented_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # 如果正在记录，处理胃部检测逻辑
        if self.recording:
            # 检查是否检测到胃部
            gastric_mask = (prediction == 1)
            has_gastric = np.any(gastric_mask)
            
            # 错误规避：检查是否是早期胃部标记
            if has_gastric and self.first_gastric_time is None and self.first_ultrasound_time is not None:
                early_detection_time = current_time - self.first_ultrasound_time
                if early_detection_time < self.early_detection_threshold:
                    self.early_detection_warning = True
                    self.result_label.setText("警告：检测到早期胃部标记，请重新扫查")
            
            if has_gastric:
                # 记录胃部首次出现时间
                if self.first_gastric_time is None:
                    self.first_gastric_time = current_time
                
                # 更新最后一次胃部出现的时间
                self.last_gastric_time = current_time
                
                # 重置消失计时
                self.no_gastric_start_time = None
                
                # 计算胃部区域的面积
                gastric_area = np.sum(gastric_mask)
                self.gastric_areas.append(gastric_area)
                self.frames_with_gastric.append(frame)
                
                # 实时更新计算结果
                if len(self.gastric_areas) > 0:
                    # 计算当前帧的容积
                    current_volume = self.calculate_gastric_volume(gastric_area)
                    # 计算平均容积
                    avg_volume = self.calculate_gastric_volume(np.mean(self.gastric_areas))
                    
                    if self.early_detection_warning:
                        self.result_label.setText("警告：检测到早期胃部标记，请重新扫查")
                    else:
                        frame_ratio = len(self.gastric_areas) / len(self.frame_timestamps)
                        result_text = f"当前容积: {current_volume:.2f} ml\n"
                        result_text += f"平均容积: {avg_volume:.2f} ml\n"
                        result_text += f"当前面积: {gastric_area:.2f} 像素\n"
                        result_text += f"帧占比: {frame_ratio:.2f}"
                        self.result_label.setText(result_text)
            else:
                # 如果没有检测到胃部，但之前检测到过
                if self.first_gastric_time is not None:
                    # 如果没有开始消失计时，开始计时
                    if self.no_gastric_start_time is None:
                        self.no_gastric_start_time = current_time
                    # 如果已经消失超过阈值时间，自动停止记录
                    elif (current_time - self.no_gastric_start_time) > self.gastric_missing_duration:
                        # 胃部已经消失超过阈值时间，自动结束记录
                        self.toggle_recording()
                    
                    if not self.early_detection_warning:
                        self.result_label.setText("未检测到胃部")
    
    def capture_screenshot(self):
        # 保存当前分割结果截图
        if hasattr(self, 'original_image') and self.original_image.pixmap():
            filename, _ = QFileDialog.getSaveFileName(self, "保存截图", "", "PNG文件 (*.png);;JPEG文件 (*.jpg *.jpeg)")
            if filename:
                self.segmented_image.pixmap().save(filename)
                
    def report_error(self):
        # 实现错误报告功能
        pass
        
    def closeEvent(self, event):
        # 关闭线程
        if self.video_thread is not None and self.video_thread.running:
            self.video_thread.stop()
        super().closeEvent(event)
                
    def upload_file(self):
        """上传图片或视频文件"""
        try:
            # 如果正在录制，先停止
            if self.recording:
                self.toggle_recording()
            
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("媒体文件 (*.jpg *.jpeg *.png *.mp4 *.avi);;所有文件 (*.*)")
            file_dialog.setViewMode(QFileDialog.ViewMode.List)
            
            if file_dialog.exec():
                file_path = file_dialog.selectedFiles()[0]
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    raise FileNotFoundError("文件不存在")
                
                # 检查文件大小
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size:
                    raise ValueError(f"文件大小超过限制（最大{self.max_file_size/1024/1024}MB）")
                
                # 检查文件类型
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']:
                    raise ValueError("不支持的文件类型")
                
                # 检查ffmpeg是否可用
                if file_ext in ['.mp4', '.avi']:
                    try:
                        import subprocess
                        subprocess.run(['ffmpeg', '-version'], capture_output=True)
                    except FileNotFoundError:
                        raise ValueError("未找到ffmpeg，请安装ffmpeg以支持视频处理")
                
                self.current_file = file_path
                
                # 停止当前视频处理（如果有）
                if self.video_thread is not None and self.video_thread.running:
                    self.video_thread.stop()
                    self.video_thread.wait()  # 等待线程完全停止
                    self.record_btn.setText("开始采集")
                
                # 根据文件类型处理
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    self.process_image(file_path)
                else:  # 视频文件
                    self.process_video(file_path)
                    
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"文件上传失败：{str(e)}")
    
    def process_image(self, image_path):
        """处理单张图片"""
        try:
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError("无法读取图片文件")
            
            # 检查图片尺寸
            if frame.shape[0] > 4096 or frame.shape[1] > 4096:
                raise ValueError("图片尺寸过大")
            
            # 进行模型推理
            prediction = self.model.predict(frame)
            
            # 重置数据和状态
            temp_recording = self.recording
            self.recording = False  # 临时设置为False防止触发update_frame中的记录逻辑
            
            # 显示原始帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.original_image.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.original_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
            
            # 显示分割结果
            # 创建彩色掩码
            color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            
            # 安全地为胃部类别创建掩码
            gastric_mask = (prediction == 1)
            if np.any(gastric_mask):  # 类别 1 (胃部) - 红色
                color_mask[gastric_mask] = [255, 0, 0]  # 红色
            
            # 将掩码与原始图像混合
            alpha = 0.5
            segmented_image = cv2.addWeighted(frame_rgb, 1-alpha, color_mask, alpha, 0)
            
            # 转换为Qt图像并显示
            h, w, ch = segmented_image.shape
            bytes_per_line = ch * w
            qt_seg_image = QImage(segmented_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.segmented_image.setPixmap(QPixmap.fromImage(qt_seg_image).scaled(
                self.segmented_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
            
            # 重置收集数据
            self.frames_with_gastric = [frame]
            self.gastric_areas = [np.sum(gastric_mask)]
            self.frame_timestamps = [time.time()]
            
            # 计算容积
            volume = self.calculate_gastric_volume(np.sum(gastric_mask))
            
            # 更新结果显示
            result_text = f"胃内容积: {volume:.2f} ml\n"
            result_text += f"当前面积: {np.sum(gastric_mask):.2f} 像素"
            self.result_label.setText(result_text)
            
            # 恢复录制状态
            self.recording = temp_recording
            self.status_label.setText("图片已处理")
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"图片处理失败：{str(e)}")
            
    def process_video(self, video_path):
        """处理视频文件"""
        try:
            # 重置数据
            self.frames_with_gastric = []
            self.gastric_areas = []
            self.result_label.setText("处理中...")
            
            # 创建新的视频线程
            self.video_thread = VideoThread(self.model, video_path)
            self.video_thread.update_frame.connect(self.update_frame)
            self.video_thread.error_occurred.connect(self.handle_video_error)
            
            # 自动开始记录
            self.recording = True
            
            # 启动线程
            self.video_thread.start()
            self.record_btn.setText("停止采集")
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"视频处理失败：{str(e)}")
    
    def handle_video_error(self, error_msg):
        """处理视频错误"""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "错误", f"视频处理出错：{error_msg}")
        if self.video_thread is not None:
            self.video_thread.stop()
        self.record_btn.setText("开始采集")

# 应用程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec()) 