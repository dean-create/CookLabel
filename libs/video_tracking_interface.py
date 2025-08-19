import sys
import os
import cv2
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

# 添加项目根目录到Python路径，使其可以单独运行
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from libs.video_auto_annotation import VideoAutoAnnotator
except ImportError:
    VideoAutoAnnotator = None


class VideoDisplay(QLabel):
    """视频显示控件"""
    frame_clicked = pyqtSignal(int, int)  # 点击位置信号
    
    def __init__(self):
        super().__init__()
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()
        
        # 设置视频显示区域的尺寸 - 高度占屏幕2/3，宽度占屏幕3/4
        video_width = int(screen_width * 0.75)  # 宽度占屏幕3/4
        video_height = int(screen_height * 0.67)  # 高度占屏幕2/3
        
        self.setMinimumSize(video_width, video_height)
        self.setMaximumSize(video_width, video_height)  # 固定尺寸
        self.setStyleSheet("border: 1px solid #ccc; background-color: black;")
        self.setAlignment(Qt.AlignCenter)  # 保持居中对齐，视频居中显示
        self.setText("请选择视频文件")
        
        # 视频相关
        self.current_frame = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 标注框
        self.manual_boxes = []  # 手动标注的框
        self.tracking_boxes = []  # 追踪的框
        self.drawing = False
        self.start_point = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_frame is not None:
            # 转换屏幕坐标到图像坐标
            x = int((event.x() - self.offset_x) / self.scale_factor)
            y = int((event.y() - self.offset_y) / self.scale_factor)
            
            self.drawing = True
            self.start_point = (x, y)
            
    def mouseMoveEvent(self, event):
        if self.drawing and self.current_frame is not None:
            # 实时显示绘制的框
            self.update_display()
            
    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton and self.current_frame is not None:
            # 先重置绘制状态，避免重复绘制
            self.drawing = False
            start_point = self.start_point
            self.start_point = None
            
            # 转换屏幕坐标到图像坐标
            end_x = int((event.x() - self.offset_x) / self.scale_factor)
            end_y = int((event.y() - self.offset_y) / self.scale_factor)
            
            if start_point:
                x1, y1 = start_point
                x2, y2 = end_x, end_y
                
                # 确保坐标正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 添加手动标注框
                if x2 - x1 > 10 and y2 - y1 > 10:  # 最小尺寸限制
                    # 弹出标签命名对话框 - 使用与主界面一致的LabelDialog
                    from libs.labelDialog import LabelDialog
                    
                    # 创建标签对话框，传入空的标签历史列表（可以根据需要扩展）
                    dialog = LabelDialog(parent=self.parent(), list_item=[])
                    # 使用pop_up方法显示对话框并获取标签名称
                    label_name = dialog.pop_up(text='')
                    
                    # 如果用户输入了标签名称（没有取消）
                    if label_name is not None and label_name.strip():
                        # 将标注框信息保存为字典格式，包含坐标和标签名称
                        box_info = {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'label': label_name.strip(),  # 用于显示的标签名称
                            'class_id': len(self.manual_boxes),  # 简单的ID分配
                            'class_name': label_name.strip()  # 用于追踪的类别名称
                        }
                        self.manual_boxes.append(box_info)
                    # 如果用户取消了标签输入，则不添加标注框
                    
            # 无论是否添加标注框，都要更新显示以清除临时绘制的框
            self.update_display()
            
    def set_frame(self, frame):
        """设置当前帧"""
        self.current_frame = frame.copy()
        self.update_display()
        
    def update_display(self):
        """更新显示"""
        if self.current_frame is None:
            return
            
        # 复制帧用于绘制
        display_frame = self.current_frame.copy()
        
        # 处理追踪结果：更新手动标注框位置而不是绘制新的追踪框
        if self.tracking_boxes and self.manual_boxes:
            # 如果有追踪结果，更新手动标注框的位置
            for i, tracking_box in enumerate(self.tracking_boxes):
                if i < len(self.manual_boxes):  # 确保不超出手动标注框的数量
                    if isinstance(tracking_box, dict):
                        # 更新手动标注框的坐标
                        self.manual_boxes[i]['x1'] = tracking_box['x1']
                        self.manual_boxes[i]['y1'] = tracking_box['y1']
                        self.manual_boxes[i]['x2'] = tracking_box['x2']
                        self.manual_boxes[i]['y2'] = tracking_box['y2']
                        
                        # 更新标签显示，包含置信度
                        original_label = self.manual_boxes[i].get('label', 'object')
                        confidence = tracking_box.get('confidence', 1.0)
                        # 只显示原始标签名称和置信度，不显示ID
                        self.manual_boxes[i]['display_label'] = f"{original_label} {confidence:.2f}"
        
        # 重新绘制更新后的手动标注框（绿色）
        for box in self.manual_boxes:
            if isinstance(box, dict):
                # 新格式：字典包含坐标和标签信息
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                # 优先使用display_label（包含置信度），否则使用原始label
                label_text = box.get('display_label', box.get('label', 'manual'))
            else:
                # 旧格式：元组只包含坐标
                x1, y1, x2, y2 = box
                label_text = "manual"
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # 绘制正在绘制的框
        if self.drawing and self.start_point:
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            end_x = int((cursor_pos.x() - self.offset_x) / self.scale_factor)
            end_y = int((cursor_pos.y() - self.offset_y) / self.scale_factor)
            
            x1, y1 = self.start_point
            cv2.rectangle(display_frame, (x1, y1), (end_x, end_y), (0, 255, 255), 2)
            
        # 转换为QPixmap并显示
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # 计算缩放比例
        widget_size = self.size()
        image_size = q_image.size()
        
        scale_x = widget_size.width() / image_size.width()
        scale_y = widget_size.height() / image_size.height()
        self.scale_factor = min(scale_x, scale_y)
        
        # 缩放图像
        scaled_size = image_size * self.scale_factor
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 计算偏移量（居中显示）
        self.offset_x = (widget_size.width() - scaled_size.width()) // 2
        self.offset_y = (widget_size.height() - scaled_size.height()) // 2
        
        self.setPixmap(scaled_pixmap)
        
    def clear_manual_boxes(self):
        """清除手动标注框"""
        self.manual_boxes.clear()
        self.update_display()
        
    def set_tracking_boxes(self, boxes):
        """设置追踪框"""
        self.tracking_boxes = boxes
        self.update_display()


class VideoTrackingInterface(QMainWindow):
    """视频追踪界面"""
    
    def __init__(self, config=None):
        """
        初始化视频追踪界面
        
        Args:
            config (dict): 配置参数字典，来自video_tracking_dialog.py对话框
                          包含: video_path, model_type, custom_model_path, frame_interval, output_dir
        """
        super().__init__()
        
        # 配置参数 - 来自对话框的配置
        self.config = config or {}
        
        # 确保输出目录固定为Auto_dataset
        if self.config:
            self.config['output_dir'] = 'Auto_dataset'
            
        self.video_path = None
        self.cap = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.is_tracking = False
        
        self.init_ui()
        
        # 如果配置中有视频路径，自动加载
        if 'video_path' in self.config:
            self.load_video(self.config['video_path'])
            
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("视频目标追踪取图工具")
        
        # 设置窗口图标（如果有的话）
        try:
            from PyQt5.QtGui import QIcon
            icon_path = os.path.join(os.path.dirname(__file__), "..", "resources", "icons", "app.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except:
            pass
        
        # 设置窗口标志，确保有完整的窗口控件（最小化、最大化、关闭按钮）
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 30, 10, 10)  # 增加上边距，让视频区域往下移动
        main_layout.setSpacing(15)  # 适当减少间距，给视频区域更多空间
        
        # 左侧视频显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 10, 5, 5)  # 调整内边距：适当增加左边距，保持平衡
        
        # 视频显示
        self.video_display = VideoDisplay()
        left_layout.addWidget(self.video_display)
        
        # 视频控制栏 - 第一行：基本播放控制
        control_layout = QHBoxLayout()
        
        self.load_video_btn = QPushButton("选择视频")
        self.load_video_btn.clicked.connect(self.select_video)
        control_layout.addWidget(self.load_video_btn)
        
        self.play_pause_btn = QPushButton("播放")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        control_layout.addWidget(self.play_pause_btn)
        
        # 进度滑块
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.valueChanged.connect(self.seek_frame)
        control_layout.addWidget(self.progress_slider)
        
        # 帧信息
        self.frame_info_label = QLabel("0 / 0")
        control_layout.addWidget(self.frame_info_label)
        
        left_layout.addLayout(control_layout)
        
        # 手动标注控制组框 - 使用QGroupBox创建灰色线框
        manual_annotation_group = QGroupBox("手动标注控制")
        manual_annotation_layout = QVBoxLayout(manual_annotation_group)
        
        # 说明文字 - 蓝色显示，字体与标题一致
        instruction_label = QLabel("说明：在视频画面上拖拽鼠标框选目标，输入标签名称，然后点击开始自动追踪")
        instruction_label.setStyleSheet("color: #2196F3; font-size: 25px; font-weight: bold;")
        instruction_label.setWordWrap(True)
        instruction_label.setAlignment(Qt.AlignCenter)  # 文字居中对齐
        manual_annotation_layout.addWidget(instruction_label)
        
        # 功能按钮栏 - 居中布局
        function_layout = QHBoxLayout()
        
        # 添加左侧弹性空间，使按钮居中
        function_layout.addStretch()
        
        # 清除标注框按钮
        self.clear_boxes_btn = QPushButton("清除标注框")
        self.clear_boxes_btn.clicked.connect(self.clear_manual_boxes)
        self.clear_boxes_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        function_layout.addWidget(self.clear_boxes_btn)
        
        # 按钮之间的间距
        function_layout.addSpacing(10)
        
        # 自动追踪按钮
        self.auto_track_btn = QPushButton("开始自动追踪")
        self.auto_track_btn.clicked.connect(self.start_auto_tracking)
        self.auto_track_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        function_layout.addWidget(self.auto_track_btn)
        
        # 添加右侧弹性空间，使按钮居中
        function_layout.addStretch()
        
        manual_annotation_layout.addLayout(function_layout)
        
        # 将手动标注组框添加到左侧布局
        left_layout.addWidget(manual_annotation_group)
        
        main_layout.addWidget(left_widget, 3)  # 左侧占3/4比例
        
        # 右侧控制面板 - 宽度占屏幕的1/4
        right_widget = QWidget()
        screen = QApplication.primaryScreen()
        screen_width = screen.size().width()
        right_panel_width = int(screen_width * 0.25)  # 宽度占屏幕1/4
        right_widget.setFixedWidth(right_panel_width)
        right_layout = QVBoxLayout(right_widget)
        
        # 配置信息
        config_group = QGroupBox("配置信息")
        config_layout = QVBoxLayout(config_group)
        
        self.config_info_label = QLabel()
        self.update_config_display()
        config_layout.addWidget(self.config_info_label)
        
        right_layout.addWidget(config_group)
        
        # 处理进度
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)
        
        # 进度条
        self.tracking_progress = QProgressBar()
        self.tracking_progress.setVisible(False)  # 初始隐藏
        progress_layout.addWidget(self.tracking_progress)
        
        # 进度状态标签
        self.progress_status_label = QLabel("等待开始追踪...")
        progress_layout.addWidget(self.progress_status_label)
        
        right_layout.addWidget(progress_group)
        
        # 处理日志 - 占满剩余空间
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        
        self.process_log = QListWidget()
        # 移除最大高度限制，让日志区域自动扩展占满剩余空间
        log_layout.addWidget(self.process_log)
        
        # 让处理日志组占据剩余的所有空间
        right_layout.addWidget(log_group, 1)  # 设置拉伸因子为1，占满剩余空间
        
        # 移除addStretch()，让处理日志区域占满底部空间
        
        main_layout.addWidget(right_widget, 1)  # 右侧占1/4比例
        
        # 定时器用于视频播放
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # 设置窗口大小和显示状态
        # 设置最小窗口大小，确保界面可用性
        self.setMinimumSize(1000, 700)
        
        # 设置窗口状态为最大化显示
        self.setWindowState(Qt.WindowMaximized)
        
        # 确保窗口在前台显示
        self.activateWindow()
        self.raise_()
        
        # 设置窗口关闭时的行为
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        try:
            # 停止定时器
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
            
            # 停止追踪工作线程
            if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(3000)  # 等待最多3秒
            
            # 释放视频资源
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            
            # 接受关闭事件
            event.accept()
            
        except Exception as e:
            print(f"关闭窗口时出错: {e}")
            event.accept()
        
    def update_config_display(self):
        """更新配置显示 - 只显示四个关键信息"""
        if not self.config:
            self.config_info_label.setText("无配置信息")
            return
        
        # 获取视频文件名（优先显示当前加载的视频，否则显示配置中的视频）
        video_name = "未选择"
        if self.video_path:
            video_name = os.path.basename(self.video_path)
        elif self.config.get('video_path'):
            video_name = os.path.basename(self.config.get('video_path'))
            
        # 获取模型类型（从对话框配置中获取）
        model_type = "未配置"
        if self.config.get('model_type') == 'custom':
            # 自定义模型，显示自定义模型文件名
            custom_model_path = self.config.get('custom_model_path', '')
            if custom_model_path:
                model_type = os.path.basename(custom_model_path)
            else:
                model_type = "自定义模型(未选择)"
        else:
            # 官方预训练模型
            model_type = self.config.get('model_type', '未配置')
                
        # 获取帧间隔
        frame_interval = self.config.get('frame_interval', 5)
        
        # 固定输出目录
        output_dir = 'Auto_dataset'
            
        config_text = f"""视频文件: {video_name}
模型类型: {model_type}
帧间隔: {frame_interval} 帧
输出目录: {output_dir}"""
        
        self.config_info_label.setText(config_text)
        

    def select_video(self):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.load_video(file_path)
            
    def load_video(self, video_path):
        """加载视频"""
        if not os.path.exists(video_path):
            return False
            
        self.video_path = video_path
        
        # 释放之前的视频
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            return False
            
        # 获取视频信息
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        
        # 更新界面
        self.progress_slider.setMaximum(self.total_frames - 1)
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(True)
        self.play_pause_btn.setEnabled(True)
        
        self.update_frame_info()
        
        # 显示第一帧
        self.show_frame(0)
        
        # 更新配置显示，显示新加载的视频文件名
        self.update_config_display()
        
        return True
        
    def show_frame(self, frame_index):
        """显示指定帧"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame_index = frame_index
            self.video_display.set_frame(frame)
            self.progress_slider.setValue(frame_index)
            self.update_frame_info()
            
    def update_frame_info(self):
        """更新帧信息"""
        self.frame_info_label.setText(f"{self.current_frame_index + 1} / {self.total_frames}")
        
    def seek_frame(self, frame_index):
        """跳转到指定帧"""
        if not self.timer.isActive():  # 只在非播放状态下响应
            self.show_frame(frame_index)
            
    def toggle_play_pause(self):
        """播放/暂停切换"""
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_btn.setText("播放")
        else:
            self.timer.start(33)  # 约30fps
            self.play_pause_btn.setText("暂停")
            
    def next_frame(self):
        """下一帧"""
        if self.current_frame_index < self.total_frames - 1:
            self.show_frame(self.current_frame_index + 1)
        else:
            self.timer.stop()
            self.play_pause_btn.setText("播放")
            
    def clear_manual_boxes(self):
        """清除手动标注框"""
        self.video_display.clear_manual_boxes()
        
            
    def start_auto_tracking(self):
        """开始自动追踪"""
        if VideoAutoAnnotator is None:
            QMessageBox.warning(self, "警告", "追踪模块未加载，无法进行自动追踪")
            return
            
        # 检查视频是否加载
        if not self.cap:
            QMessageBox.warning(self, "警告", "请先加载视频文件")
            return
            
        # 检查模型配置 - 检查model_type或custom_model_path
        model_configured = False
        if self.config.get('model_type') == 'custom':
            # 自定义模型，检查custom_model_path
            if self.config.get('custom_model_path'):
                model_configured = True
        elif self.config.get('model_type'):
            # 官方预训练模型，检查model_type
            model_configured = True
            
        if not model_configured:
            QMessageBox.warning(self, "警告", "请先配置模型类型")
            return
            
        if not self.config.get('output_dir'):
            QMessageBox.warning(self, "警告", "请配置输出目录")
            return
            
        # 检查手动标注框
        if not self.video_display.manual_boxes:
            # 没有手动标注框，询问是否继续
            reply = QMessageBox.question(
                self, "确认", 
                "未发现手动标注框，是否继续自动追踪？\n"
                "建议先进行手动标注以提高追踪精度。",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 直接使用已包含标签信息的手动标注框
        manual_boxes_with_classes = self.video_display.manual_boxes
        
        # 初始化进度显示
        self.tracking_progress.setVisible(True)
        self.tracking_progress.setValue(0)
        self.tracking_progress.setMaximum(100)
        self.progress_status_label.setText("准备开始追踪...")
        
        # 清空处理日志
        self.process_log.clear()
        self.process_log.addItem("开始自动追踪...")
        
        # 停止当前视频播放（如果正在播放）
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_btn.setText("播放")
        
        # 更改按钮状态
        self.auto_track_btn.setText("停止追踪")
        self.auto_track_btn.clicked.disconnect()
        self.auto_track_btn.clicked.connect(self.stop_auto_tracking)
        self.auto_track_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        # 禁用播放控制，防止用户在追踪过程中手动控制
        self.play_pause_btn.setEnabled(False)
        self.progress_slider.setEnabled(False)
        
        self.is_tracking = True
        
        # 创建追踪工作线程
        self.tracking_worker = VideoTrackingWorker(
            self.config, 
            manual_boxes_with_classes,
            self.current_frame_index
        )
        
        # 连接信号
        self.tracking_worker.progress_updated.connect(self.update_tracking_progress)
        self.tracking_worker.frame_processed.connect(self.on_frame_processed)
        self.tracking_worker.tracking_finished.connect(self.on_tracking_finished)
        self.tracking_worker.statistics_ready.connect(self.on_statistics_ready)
        
        # 启动线程
        self.tracking_worker.start()
        
    def stop_auto_tracking(self):
        """停止自动追踪"""
        if hasattr(self, 'tracking_worker'):
            self.tracking_worker.stop()
            
        self.is_tracking = False
        
        # 重置进度显示
        self.progress_status_label.setText("追踪已停止")
        self.process_log.addItem("用户停止追踪")
        
        # 恢复播放控制
        self.play_pause_btn.setEnabled(True)
        self.progress_slider.setEnabled(True)
        
        # 恢复按钮状态
        self.auto_track_btn.setText("开始自动追踪")
        self.auto_track_btn.clicked.disconnect()
        self.auto_track_btn.clicked.connect(self.start_auto_tracking)
        self.auto_track_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
    def on_progress_updated(self, frame_index):
        """进度更新回调"""
        # 更新进度条
        if hasattr(self, 'total_frames') and self.total_frames > 0:
            progress_percent = int((frame_index / self.total_frames) * 100)
            self.tracking_progress.setValue(progress_percent)
            
        # 更新状态标签
        self.progress_status_label.setText(f"正在处理第 {frame_index + 1} 帧...")
        
        # 添加到处理日志
        item_text = f"处理第 {frame_index + 1} 帧"
        self.process_log.addItem(item_text)
        self.process_log.scrollToBottom()
        
    def on_frame_processed(self, frame_index, results):
        """帧处理完成回调"""
        # 自动跳转到当前处理的帧，实现视频播放效果
        if frame_index != self.current_frame_index:
            self.show_frame(frame_index)
        
        if results:
            # 转换追踪结果为显示格式，更新手动标注框的位置和标签
            tracking_boxes = []
            for result in results:
                # 确保使用正确的属性名
                tracking_boxes.append({
                    'x1': int(result.x1),
                    'y1': int(result.y1), 
                    'x2': int(result.x2),
                    'y2': int(result.y2),
                    'track_id': result.track_id,
                    'class_name': result.class_name,
                    'confidence': result.confidence
                })
            
            # 更新追踪框显示，保持绿色标注框可见
            self.video_display.set_tracking_boxes(tracking_boxes)
                
            # 添加详细的处理日志，包含置信度信息
            for result in results:
                log_text = f"第 {frame_index + 1} 帧追踪到目标: {result.class_name} (置信度: {result.confidence:.2f})"
                self.process_log.addItem(log_text)
            self.process_log.scrollToBottom()
        else:
            # 未检测到目标时清除追踪框显示
            self.video_display.set_tracking_boxes([])
                
            # 未检测到目标的日志
            log_text = f"第 {frame_index + 1} 帧未检测到目标"
            self.process_log.addItem(log_text)
            self.process_log.scrollToBottom()
                
    def update_tracking_progress(self, frame_index):
        """更新追踪进度"""
        if hasattr(self, 'total_frames') and self.total_frames > 0:
            progress = int((frame_index / self.total_frames) * 100)
            self.tracking_progress.setValue(progress)
            self.progress_status_label.setText(f"正在追踪... {progress}%")
    
    def on_statistics_ready(self, stats, images_dir, labels_dir):
        """处理统计信息"""
        # 在日志中输出统计信息
        self.process_log.addItem("=" * 50)
        self.process_log.addItem("处理完成！统计信息：")
        
        # 格式化统计信息
        stats_text = "{"
        for key, value in stats.items():
            if key in ['total_frames', 'total_tracks', 'active_tracks', 'total_detections']:
                stats_text += f"'{key}': {value}, "
        stats_text = stats_text.rstrip(", ") + "}"
        
        self.process_log.addItem(stats_text)
        
        # 输出保存路径信息
        saved_images = stats.get('saved_images', 0)
        saved_labels = stats.get('saved_labels', 0)
        
        self.process_log.addItem(f"共保存 {saved_images} 组完整原图到：{images_dir}")
        self.process_log.addItem(f"共保存 {saved_labels} 组YOLO标签到：{labels_dir}")
        
        self.process_log.scrollToBottom()
    
    def on_tracking_finished(self, success):
        """追踪完成回调"""
        self.is_tracking = False
        
        # 更新进度显示
        if success:
            self.tracking_progress.setValue(100)
            self.progress_status_label.setText("追踪完成！")
            # 不在这里添加"追踪完成！"，因为统计信息会在之后显示
        else:
            self.progress_status_label.setText("追踪失败！")
            self.process_log.addItem("追踪失败！")
        
        # 恢复播放控制
        self.play_pause_btn.setEnabled(True)
        self.progress_slider.setEnabled(True)
        
        # 恢复按钮状态
        self.auto_track_btn.setText("开始自动追踪")
        self.auto_track_btn.clicked.disconnect()
        self.auto_track_btn.clicked.connect(self.start_auto_tracking)
        self.auto_track_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        if success:
            # 显示结果对话框
            video_path = self.config.get('video_path', '')
            import os
            video_dir = os.path.dirname(video_path)
            output_dir = os.path.join(video_dir, 'Auto_dataset')
            images_dir = os.path.join(output_dir, 'images')
            labels_dir = os.path.join(output_dir, 'labels')
            
            QMessageBox.information(
                self, 
                "追踪完成", 
                f"视频追踪已完成！\n\n"
                f"输出目录: {output_dir}\n\n"
                f"生成的文件结构:\n"
                f"📁 Auto_dataset/\n"
                f"  ├── 📁 images/     (原始帧图像)\n"
                f"  ├── 📁 labels/     (YOLO格式标注文件)\n"
                f"  └── 📄 tracking_results.json  (追踪结果摘要)\n\n"
                f"详细路径:\n"
                f"• 图像文件: {images_dir}\n"
                f"• 标注文件: {labels_dir}\n"
                f"• 结果文件: {os.path.join(output_dir, 'tracking_results.json')}"
            )
        else:
            QMessageBox.warning(self, "追踪失败", "视频追踪过程中出现错误，请查看控制台日志。")
            
    def closeEvent(self, event):
        """关闭事件"""
        if self.cap:
            self.cap.release()
            
        if hasattr(self, 'tracking_worker') and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
            self.tracking_worker.wait()
            
        event.accept()


class VideoTrackingWorker(QThread):
    """视频追踪工作线程"""
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(int, list)
    tracking_finished = pyqtSignal(bool)
    statistics_ready = pyqtSignal(dict, str, str)  # 统计信息, 图片目录, 标签目录
    
    def __init__(self, config, manual_boxes, start_frame=0):
        super().__init__()
        self.config = config
        self.manual_boxes = manual_boxes
        self.start_frame = start_frame
        self.should_stop = False
        
    def stop(self):
        """停止追踪"""
        self.should_stop = True
        
    def run(self):
        """运行追踪"""
        try:
            # 检查追踪器是否可用
            if VideoAutoAnnotator is None:
                self.tracking_finished.emit(False)
                return
                
            # 获取配置参数
            video_path = self.config.get('video_path', '')
            frame_interval = self.config.get('frame_interval', 5)
            model_path = self.config.get('custom_model_path', None)
            if not model_path or model_path.strip() == '':
                model_path = None  # 使用默认模型
                
            # 将输出目录设置为视频文件的同目录
            import os
            video_dir = os.path.dirname(video_path)
            output_dir = os.path.join(video_dir, 'Auto_dataset')
            
            # 创建标准的数据集目录结构
            images_dir = os.path.join(output_dir, 'images')
            labels_dir = os.path.join(output_dir, 'labels')
            
            # 确保目录存在
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
                
            # 创建追踪器，输出目录设置为Auto_dataset根目录
            tracker = VideoAutoAnnotator(
                output_dir=output_dir,
                model_path=model_path,
                confidence_threshold=0.5,
                high_iou_threshold=0.3,  # 降低高IoU阈值，提高追踪鲁棒性
                low_iou_threshold=0.15,  # 降低低IoU阈值，提高追踪鲁棒性
                max_disappeared=30
            )
            
            # 打开视频文件
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                self.tracking_finished.emit(False)
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            
            # 读取第一帧并初始化追踪器
            ret, first_frame = cap.read()
            if not ret:
                print("无法读取第一帧")
                cap.release()
                self.tracking_finished.emit(False)
                return
                
            # 处理手动标注框格式
            manual_boxes_dict = []
            if self.manual_boxes:
                # 检查是否已经是带类别信息的格式
                if isinstance(self.manual_boxes[0], dict):
                    # 已经是带类别信息的格式，直接使用
                    manual_boxes_dict = self.manual_boxes
                else:
                    # 旧格式，转换为带默认类别信息的格式
                    for i, (x1, y1, x2, y2) in enumerate(self.manual_boxes):
                        manual_boxes_dict.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'class_id': 0,  # 默认类别ID
                            'class_name': 'object'  # 默认类别名称
                        })
            
            # 初始化追踪器
            first_results = tracker.initialize_from_manual_annotation(first_frame, manual_boxes_dict)
            
            # 保存第一帧图片和标签
            saved_count = 0
            frame_name = f"frame_{saved_count:06d}"
            image_path = os.path.join(images_dir, f"{frame_name}.jpg")
            cv2.imwrite(image_path, first_frame)
            
            # 手动更新图片统计计数
            tracker.statistics['saved_images'] += 1
            
            # 保存标注文件，并根据保存结果更新统计计数
            height, width = first_frame.shape[:2]
            annotation_saved = tracker.annotation_generator.save_frame_annotation(
                frame_name, first_results, width, height
            )
            
            # 只有成功保存标注文件时才更新计数
            if annotation_saved:
                tracker.statistics['saved_annotations'] += 1
                tracker.statistics['saved_labels'] += 1
            
            # 发送第一帧的结果
            self.progress_updated.emit(self.start_frame)
            self.frame_processed.emit(self.start_frame, first_results)
            saved_count += 1
            
            # 处理后续帧
            frame_index = self.start_frame + 1
            while frame_index < total_frames and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 判断是否需要保存当前帧（按间隔）
                should_save = (frame_index - self.start_frame) % frame_interval == 0
                
                if should_save:
                    # 生成文件名
                    frame_name = f"frame_{saved_count:06d}"
                    image_path = os.path.join(images_dir, f"{frame_name}.jpg")
                    
                    # 处理当前帧（明确要求保存标注）
                    results = tracker.process_frame(frame, frame_name, save_annotation=True)
                    
                    # 保存图片到images目录
                    cv2.imwrite(image_path, frame)
                    
                    # 手动更新图片统计计数
                    tracker.statistics['saved_images'] += 1
                    
                    saved_count += 1
                else:
                    # 只处理追踪，不保存文件
                    results = tracker.process_frame(frame, save_annotation=False)
                
                # 无论是否保存，都发送进度和结果（用于实时显示）
                self.progress_updated.emit(frame_index)
                self.frame_processed.emit(frame_index, results)
                
                # 添加小延时，控制播放速度，避免过快
                import time
                time.sleep(0.03)  # 约33fps的播放速度
                
                frame_index += 1
                
            cap.release()
            
            # 导出追踪结果
            results_file = os.path.join(output_dir, "tracking_results.json")
            tracker.export_tracking_results(results_file)
            
            # 获取统计信息
            stats = tracker.get_tracking_statistics()
            print(f"追踪完成: {stats}")
            print(f"实际保存图片数量: {saved_count} 张")
            
            # 发送统计信息
            self.statistics_ready.emit(stats, images_dir, labels_dir)
            
            self.tracking_finished.emit(not self.should_stop)
            
        except Exception as e:
            print(f"追踪过程中出错: {e}")
            import traceback
            traceback.print_exc()
            self.tracking_finished.emit(False)


def main(config=None):
    """
    主函数 - 可以独立运行或从对话框调用
    
    Args:
        config (dict): 来自video_tracking_dialog.py的配置参数
                      包含: video_path, model_type, custom_model_path, frame_interval等
    """
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("视频目标追踪工具")
    app.setApplicationVersion("1.0")
    
    # 如果没有传入配置，使用默认配置（用于独立测试）
    if config is None:
        config = {
            'video_path': '',
            'model_type': 'yolov8n',
            'custom_model_path': '',
            'frame_interval': 5,
            'output_dir': 'Auto_dataset'
        }
    
    # 创建主窗口
    window = VideoTrackingInterface(config)
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()