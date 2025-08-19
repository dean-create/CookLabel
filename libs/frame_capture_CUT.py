import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
import torch
import re
from efficientnet_pytorch import EfficientNet

import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)

# 模型类别配置
class_names = {
    0: 'no',
    1: 'yes',
}

class_colors = {
    'no': (255, 255, 255),  # 白色
    'yes': (0, 255, 0),          # 绿色
}

class ModelLoader:
    def __init__(self, model_type, model_path, num_classes, name=None):
        self.model_type = model_type
        self.model_path = model_path
        self.num_classes = num_classes
        self.name = name or os.path.basename(model_path)  # 使用文件名作为默认名称
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == 'YOLO':
            try:
                return YOLO(self.model_path)
            except FileNotFoundError:
                print(f"检测模型文件不存在，请检查路径：{self.model_path}")
                return None
        elif self.model_type == 'EfficientNet':
            try:

                model = EfficientNet.from_name('efficientnet-b0', num_classes=self.num_classes)
                
                num_ftrs = model._fc.in_features
                model._fc = MLPBlock(input_dim=num_ftrs, hidden_dim=256, output_dim=2)

                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                return model
            except FileNotFoundError:
                print(f"分类模型文件不存在，请检查路径：{self.model_path}")
                return None
        else:
            print(f"不支持的模型类型：{self.model_type}")
            return None

class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分类模型管理")
        self.setGeometry(200, 200, 600, 400)
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SingleSelection)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("已加载的分类模型:"))
        layout.addWidget(self.model_list)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加模型")
        self.add_btn.clicked.connect(self.add_model)
        btn_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("删除模型")
        self.remove_btn.clicked.connect(self.remove_model)
        btn_layout.addWidget(self.remove_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def add_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择分类模型文件", 
            "", 
            "模型文件 (*.pth *.tar);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                model = EfficientNet.from_name('efficientnet-b0', num_classes=2)

                num_ftrs = model._fc.in_features
                model._fc = MLPBlock(input_dim=num_ftrs, hidden_dim=256, output_dim=2)

                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
                
                # 检查状态字典格式
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                model.load_state_dict(state_dict)
                model.eval()
                
                model_name = os.path.basename(file_path)
                self.model_list.addItem(model_name)
                self.parent().add_classification_model(model, model_name)
                
                QMessageBox.information(self, "成功", f"分类模型已加载: {model_name}")
            except Exception as e:
                QMessageBox.warning(self, "模型加载错误", f"无法加载分类模型: {str(e)}")
    
    def remove_model(self):
        selected = self.model_list.currentRow()
        if selected >= 0:
            model_name = self.model_list.item(selected).text()
            self.model_list.takeItem(selected)
            self.parent().remove_classification_model(model_name)
    
    def set_models(self, models):
        self.model_list.clear()
        for model_info in models:
            self.model_list.addItem(model_info['name'])

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频帧捕捉工具")
        # 设置窗口默认最大化显示，提供更好的视频查看体验
        self.showMaximized()

        # 新增变量0716
        self.locked_box_index = None # 锁定的框索引，None表示未锁定
        self.box_order = [] # 按从左到右排序的框列表

        # 视频处理相关变量
        self.video_folder = ""
        self.video_files = []
        self.current_video_index = -1
        self.cap = None
        self.playing = False
        self.play_speed = 1.0
        self.current_frame_pos = 0
        self.total_frames = 0
        self.fps = 30
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        
        # 模型推理相关变量
        self.model_loaded = False
        self.detection_model = None
        self.classification_models = []
        self.inference_results = None
        self.inference_active = False  # 新增：推理状态标志
        self.original_frame = None  # 新增：存储原始帧用于保存
        
        # UI相关变量
        self.frame_info_label = QLabel()
        self.model_status_label = QLabel("模型未加载")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("color: #FF5252; font-size: 12px;")
        self.is_seeking = False
        self.save_dir = os.path.abspath(r"D:\project\ultralytics-main\saved_frames") # 添加保存目录
        
        # 按钮高亮状态
        self.active_button = None
        self.button_highlight_timer = QTimer(self)
        self.button_highlight_timer.timeout.connect(self.clear_button_highlight)
        self.button_highlight_duration = 300  # 高亮持续时间(毫秒)

        self.init_ui()
        
        # 尝试加载模型
        self.load_models()

    def init_ui(self):
        # 创建主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制区域
        top_layout = QHBoxLayout()

        # 加载文件夹按钮
        self.load_folder_btn = QPushButton("加载视频文件夹")
        self.load_folder_btn.clicked.connect(self.load_video_folder)
        self.load_folder_btn.setFixedHeight(40)
        top_layout.addWidget(self.load_folder_btn)

        # 视频文件下拉框
        self.video_combo = QComboBox()
        self.video_combo.setFixedHeight(40)
        self.video_combo.currentIndexChanged.connect(self.select_video)
        top_layout.addWidget(self.video_combo, 2)

        # 添加模型加载按钮
        self.load_model_btn = QPushButton("加载检测模型")
        self.load_model_btn.setFixedSize(200, 40)  # 设置固定宽度150像素，高度40像素
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        top_layout.addWidget(self.load_model_btn)
        
        # 添加模型管理按钮
        self.manage_models_btn = QPushButton("管理分类模型")
        self.manage_models_btn.setFixedSize(200, 40)  # 设置固定宽度150像素，高度40像素
        self.manage_models_btn.clicked.connect(self.manage_classification_models)
        top_layout.addWidget(self.manage_models_btn)

        main_layout.addLayout(top_layout)

        # 视频显示区域
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)

        main_layout.addWidget(video_container, 5)

        # 控制按钮区域 - 统一的水平布局，参考videolabeltool.py的整齐排列
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        # 导航控制组 - 上一帧和快退按钮
        self.prev_frame_btn = QPushButton("上一帧")
        self.prev_frame_btn.setFixedSize(110, 40)
        self.prev_frame_btn.clicked.connect(lambda: self.navigate_frames(-1))
        self.prev_frame_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.prev_frame_btn)

        self.prev_5_btn = QPushButton("快退5帧")
        self.prev_5_btn.setFixedSize(110, 40)
        self.prev_5_btn.clicked.connect(lambda: self.navigate_frames(-5))
        self.prev_5_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.prev_5_btn)

        # 播放控制组 - 播放按钮
        self.play_btn = QPushButton("播放")
        self.play_btn.setFixedSize(110, 40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.play_btn)

        # 倍速控制组 - 所有倍速按钮
        self.speed_05x = QPushButton("0.5x")
        self.speed_05x.setFixedSize(110, 40)
        self.speed_05x.clicked.connect(lambda: self.set_speed(0.5))
        self.speed_05x.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.speed_05x)

        self.speed_1x = QPushButton("1x")
        self.speed_1x.setFixedSize(110, 40)
        self.speed_1x.clicked.connect(lambda: self.set_speed(1.0))
        self.speed_1x.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.speed_1x)

        self.speed_2x = QPushButton("2x")
        self.speed_2x.setFixedSize(110, 40)
        self.speed_2x.clicked.connect(lambda: self.set_speed(2.0))
        self.speed_2x.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.speed_2x)

        self.speed_25x = QPushButton("3x")
        self.speed_25x.setFixedSize(110, 40)
        self.speed_25x.clicked.connect(lambda: self.set_speed(3.0))
        self.speed_25x.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.speed_25x)

        # 推理控制组 - 推理按钮（移动到3x和快进5帧之间）
        self.inference_btn = QPushButton("开始推理")
        self.inference_btn.setFixedSize(110, 40)
        self.inference_btn.setCheckable(True)  # 设置为可切换状态
        self.inference_btn.clicked.connect(self.toggle_inference)
        self.inference_btn.setEnabled(False)
        self.inference_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.inference_btn)

        # 导航控制组 - 下一帧和快进按钮
        self.next_5_btn = QPushButton("快进5帧")
        self.next_5_btn.setFixedSize(110, 40)
        self.next_5_btn.clicked.connect(lambda: self.navigate_frames(5))
        self.next_5_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.next_5_btn)

        self.next_frame_btn = QPushButton("下一帧")
        self.next_frame_btn.setFixedSize(110, 40)
        self.next_frame_btn.clicked.connect(lambda: self.navigate_frames(1))
        self.next_frame_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.next_frame_btn)

        # 将控制按钮组添加到主布局
        main_layout.addWidget(controls_widget, stretch=0)

        # 进度条
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.sliderMoved.connect(self.set_position)
        self.progress_bar.sliderPressed.connect(self.on_seek_start)
        self.progress_bar.sliderReleased.connect(self.on_seek_end)
        self.progress_bar.setEnabled(False)
        main_layout.addWidget(self.progress_bar)

        # 帧信息和模型状态
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.frame_info_label, alignment=Qt.AlignLeft)
        info_layout.addWidget(self.model_status_label, alignment=Qt.AlignRight)
        main_layout.addLayout(info_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 快捷键提示
        self.shortcut_label = QLabel(
            "快捷键: 1/2/3=锁定第1/2/3个框(从左到右) | S=保存到root文件夹 | W=保存到0文件夹 | E=保存到1文件夹 | R=保存到2文件夹 | T=保存到3文件夹 | ←/→=快退/快进 | 空格=播放/暂停 | ↑/↓=切换视频 | A/D=单帧进退")
        self.shortcut_label.setAlignment(Qt.AlignCenter)
        self.shortcut_label.setStyleSheet("font-weight: bold; color: #555; padding: 5px;")
        main_layout.addWidget(self.shortcut_label)

        # 保存帧按钮
        save_layout = QHBoxLayout()
        
        self.save_root_btn = QPushButton("保存到root文件夹 (S)")
        self.save_root_btn.setFixedSize(250, 40)
        self.save_root_btn.clicked.connect(lambda: self.save_current_frame(0))
        save_layout.addWidget(self.save_root_btn)
        
        self.save_0_btn = QPushButton("保存到0文件夹 (W)")
        self.save_0_btn.setFixedSize(250, 40)
        self.save_0_btn.clicked.connect(lambda: self.save_current_frame(1))
        save_layout.addWidget(self.save_0_btn)
        
        self.save_1_btn = QPushButton("保存到1文件夹 (E)")
        self.save_1_btn.setFixedSize(250, 40)
        self.save_1_btn.clicked.connect(lambda: self.save_current_frame(2))
        save_layout.addWidget(self.save_1_btn)
        
        self.save_2_btn = QPushButton("保存到2文件夹 (R)")
        self.save_2_btn.setFixedSize(250, 40)
        self.save_2_btn.clicked.connect(lambda: self.save_current_frame(3))
        save_layout.addWidget(self.save_2_btn)
        
        self.save_3_btn = QPushButton("保存到3文件夹 (T)")
        self.save_3_btn.setFixedSize(250, 40)
        self.save_3_btn.clicked.connect(lambda: self.save_current_frame(4))
        save_layout.addWidget(self.save_3_btn)
        
        main_layout.addLayout(save_layout)

        # 初始化帧图像变量
        self.current_frame = None
        self.original_frame = None
        
        # 记录最后保存的文件路径，用于撤回功能
        self.last_saved_file = None  # 确保初始化
        self.display_placeholder()

        # 禁用所有按钮的焦点获取
        self.disable_button_focus()

        # 初始按钮样式
        self.update_button_styles()

    def disable_button_focus(self):
        buttons = [
            self.play_btn, self.inference_btn, self.speed_05x, self.speed_1x, 
            self.speed_2x, self.speed_25x, self.prev_frame_btn, self.next_frame_btn, 
            self.prev_5_btn, self.next_5_btn, self.save_root_btn, self.save_0_btn, self.save_1_btn,
            self.save_2_btn, self.save_3_btn, self.load_model_btn, self.manage_models_btn
        ]
        for btn in buttons:
            btn.setFocusPolicy(Qt.NoFocus)

    def update_button_styles(self):
        # 播放按钮样式
        if self.playing:
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50; 
                    color: white;
                    border: 1px solid #388E3C;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #388E3C;
                }
            """)
        else:
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0; 
                    color: #333;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)

        # 推理按钮样式
        if self.inference_active:
            self.inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF5722; 
                    color: white;
                    border: 1px solid #E64A19;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #E64A19;
                }
            """)
        elif self.inference_btn.isEnabled():
            self.inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3; 
                    color: white;
                    border: 1px solid #0b7dda;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
            """)
        else:
            self.inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0; 
                    color: #333;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)

        # 模型加载按钮样式
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; 
                color: white;
                border: 1px solid #7B1FA2;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        
        # 模型管理按钮样式
        self.manage_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #009688; 
                color: white;
                border: 1px solid #00796B;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00796B;
            }
        """)

        # 导航按钮样式
        nav_buttons = [self.prev_frame_btn, self.next_frame_btn, self.prev_5_btn, self.next_5_btn]
        for btn in nav_buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #607D8B; 
                    color: white;
                    border: 1px solid #546E7A;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #546E7A;
                }
            """)

        # 保存按钮样式
        save_buttons = [self.save_root_btn, self.save_0_btn, self.save_1_btn, self.save_2_btn, self.save_3_btn]
        for btn in save_buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800; 
                    color: white;
                    border: 1px solid #F57C00;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)

        # 倍速按钮样式
        speed_buttons = [self.speed_05x, self.speed_1x, self.speed_2x, self.speed_25x]
        for btn in speed_buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3F51B5; 
                    color: white;
                    border: 1px solid #303F9F;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #303F9F;
                }
            """)

        # 高亮当前活动按钮
        if self.active_button:
            self.active_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF5722; 
                    color: white;
                    border: 2px solid #E64A19;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)

    def display_placeholder(self):
        pixmap = QPixmap(800, 400)
        pixmap.fill(QColor(30, 30, 30))
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 24))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "加载视频文件夹开始使用")
        painter.end()
        self.video_label.setPixmap(pixmap)

    def load_video_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择视频文件夹")

        if folder_path:
            self.video_folder = folder_path
            self.video_files = []
            self.video_combo.clear()

            # 扫描支持的视频文件
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            for file in os.listdir(folder_path):
                if os.path.splitext(file)[1].lower() in extensions:
                    self.video_files.append(file)

            if not self.video_files:
                QMessageBox.warning(self, "无视频文件", "该文件夹中没有找到支持的视频文件")
                return

            self.video_combo.addItems(self.video_files)
            self.status_bar.showMessage(f"已加载文件夹: {folder_path} | 共 {len(self.video_files)} 个视频")
            self.highlight_button(self.load_folder_btn)

    def select_video(self, index):
        if index < 0 or index >= len(self.video_files):
            return

        self.current_video_index = index
        video_file = os.path.join(self.video_folder, self.video_files[index])

        # 释放之前的视频资源
        if self.cap:
            self.cap.release()
            self.frame_timer.stop()

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件: {video_file}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_pos = 0
        self.playing = False
        self.play_btn.setText("播放")

        # 设置进度条范围
        self.progress_bar.setRange(0, self.total_frames - 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(True)
        
        # 启用控件
        self.play_btn.setEnabled(True)
        self.inference_btn.setEnabled(self.model_loaded)
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
        self.prev_5_btn.setEnabled(True)
        self.next_5_btn.setEnabled(True)
        self.save_root_btn.setEnabled(True)
        self.save_0_btn.setEnabled(True)
        self.save_1_btn.setEnabled(True)

        # 显示第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.original_frame = frame.copy()  # 保存原始帧
            self.display_frame(frame)
            self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
        self.status_bar.showMessage(
            f"已加载: {self.video_files[index]} | 总帧数: {self.total_frames} | FPS: {self.fps:.2f}")

    def toggle_play(self):
        if not self.cap or not self.cap.isOpened():
            return

        self.playing = not self.playing

        if self.playing:
            self.play_btn.setText("暂停")
            self.frame_timer.start(int(1000 / (self.fps * self.play_speed)))
        else:
            self.play_btn.setText("播放")
            self.frame_timer.stop()

        self.update_button_styles()
        self.setFocus()
        self.highlight_button(self.play_btn)

    def set_speed(self, speed):
        self.play_speed = speed

        # 更新当前激活的倍速按钮
        buttons = {0.5: self.speed_05x, 1.0: self.speed_1x, 2.0: self.speed_2x, 3.0: self.speed_25x}
        for s, btn in buttons.items():
            if s == speed:
                self.highlight_button(btn)
            else:
                btn.setStyleSheet("background-color: #3F51B5; color: white; border-radius: 4px;")

        if self.playing:
            self.frame_timer.start(int(1000 / (self.fps * speed)))

        self.setFocus()

    def set_position(self, position):
        if not self.cap:
            return
        position = max(0, min(position, self.total_frames - 1))
        if self.current_frame_pos == position:
            return
            
        self.current_frame_pos = position
        
        if not self.playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.original_frame = frame.copy()  # 更新原始帧
                # 如果处于推理状态，运行推理
                if self.inference_active:
                    self.run_inference(frame)
                self.display_frame(frame)
                self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
        
        self.progress_bar.setValue(self.current_frame_pos)
        self.setFocus()

    def on_seek_start(self):
        self.is_seeking = True
        if self.playing:
            self.was_playing = True
            self.toggle_play()
        else:
            self.was_playing = False

    def on_seek_end(self):
        self.is_seeking = False
        if self.was_playing:
            self.toggle_play()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            if self.playing and not self.is_seeking:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self.current_frame = frame
                    self.original_frame = frame.copy()  # 更新原始帧
                    self.progress_bar.setValue(self.current_frame_pos)
                    
                    # 如果处于推理状态，运行推理
                    if self.inference_active:
                        self.run_inference(frame)
                        
                    self.display_frame(frame)
                    self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
                else:
                    # 视频结束，重置到第一帧
                    self.current_frame_pos = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.toggle_play()

    def display_frame(self, frame):
        if frame is None:
            return

        # 如果有推理结果且处于推理状态，将其绘制到帧上
        if self.inference_active and self.inference_results is not None:
            frame = self.draw_inference_results(frame)
        
        # 新增：绘制锁定框的高亮标记
        if self.locked_box_index is not None and 0 <= self.locked_box_index < len(self.box_order):
            x1, y1, x2, y2 = self.box_order[self.locked_box_index]['box']
            # 红色双重边框表示锁定状态
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # 外框
            cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (0, 0, 255), 1) # 内框
            # 显示简洁的锁定框编号，格式为"Lock X"（X为从左到右第几个框）
            cv2.putText(frame, f"Lock {self.locked_box_index+1}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 将OpenCV图像转换为Qt图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放以适应标签大小
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """处理窗口大小调整事件，确保视频帧正确显示"""
        super().resizeEvent(event)
        # 检查current_frame属性是否存在且不为None，避免初始化时的属性错误
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.display_frame(self.current_frame)

    def keyPressEvent(self, event):
        if not self.cap:
            return super().keyPressEvent(event)

        key = event.key()
        step = 0

        # 重新规划按键映射，使用不同的值区分导航和保存功能
        key_map = {
            Qt.Key_Left: -5,
            Qt.Key_Right: 5,
            Qt.Key_A: -1,
            Qt.Key_D: 1,  # 保持导航功能
            Qt.Key_Space: None,
            Qt.Key_Up: None,
            Qt.Key_Down: None,
            Qt.Key_S: 'save_0',  # 使用字符串标记保存功能
            Qt.Key_W: 'save_1',  # 使用字符串标记保存功能
            Qt.Key_E: 'save_2',  # 使用字符串标记保存功能
            Qt.Key_R: 'save_3',  # 新增：保存到2文件夹
            Qt.Key_T: 'save_4',  # 新增：保存到3文件夹
            Qt.Key_Q: 'undo'     # 新增：撤回功能
        }
        
        action = key_map.get(key, None)

        if action is not None:
            if isinstance(action, str):  # 处理保存操作和撤回操作
                if action == 'undo':  # 处理撤回操作
                    self.undo_last_save()
                else:  # 处理保存操作
                    # 从字符串中提取保存目标文件夹编号
                    target_folder = int(action.split('_')[1])
                    self.save_current_frame(target_folder)
                    # 高亮对应的保存按钮
                    if target_folder == 0:
                        self.highlight_button(self.save_root_btn)
                    elif target_folder == 1:
                        self.highlight_button(self.save_0_btn)
                    elif target_folder == 2:
                        self.highlight_button(self.save_1_btn)
                    elif target_folder == 3:
                        self.highlight_button(self.save_2_btn)
                    elif target_folder == 4:
                        self.highlight_button(self.save_3_btn)
            else:  # 处理导航操作
                self.navigate_frames(action)
                # 高亮对应的导航按钮
                if key == Qt.Key_Left:
                    self.highlight_button(self.prev_5_btn)
                elif key == Qt.Key_Right:
                    self.highlight_button(self.next_5_btn)
                elif key == Qt.Key_A:
                    self.highlight_button(self.prev_frame_btn)
                elif key == Qt.Key_D:
                    self.highlight_button(self.next_frame_btn)
        
        # 新增：数字键1-3锁定框的功能（放在现有逻辑之后）0716
        if Qt.Key_1 <= key <= Qt.Key_3:
            box_num = key - Qt.Key_0 # 转换为数字1-3
            self.lock_box(box_num)
            self.highlight_button(None) # 清除按钮高亮
            return

        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Up and self.current_video_index > 0:
            self.video_combo.setCurrentIndex(self.current_video_index - 1)
        elif key == Qt.Key_Down and self.current_video_index < len(self.video_files) - 1:
            self.video_combo.setCurrentIndex(self.current_video_index + 1)
        else:
            super().keyPressEvent(event)

        self.setFocus()

    # 处理鼠标滚轮事件
    def wheelEvent(self, event):
        if not self.cap or not self.cap.isOpened():
            return super().wheelEvent(event)
            
        # 获取滚轮滚动方向和步数
        delta = event.angleDelta().y()
        step = 5 if delta > 0 else -5  # 向上滚动前进5帧，向下滚动后退5帧
        
        # 暂停播放并导航帧
        if self.playing:
            self.toggle_play()
            
        self.navigate_frames(step)
        
        # 高亮对应的导航按钮
        self.highlight_button(self.prev_5_btn if step < 0 else self.next_5_btn)

    def navigate_frames(self, step):
        if not self.cap:
            return
            
        # 导航时暂停播放
        if self.playing:
            self.toggle_play()
            
        new_pos = self.current_frame_pos + step
        new_pos = max(0, min(new_pos, self.total_frames - 1))
        
        if new_pos != self.current_frame_pos:
            self.set_position(new_pos)

    def toggle_inference(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "模型未加载", "请确保检测模型和分类模型文件存在于正确路径")
            self.inference_btn.setChecked(False)  # 确保按钮状态正确
            return

        # 切换推理状态
        self.inference_active = self.inference_btn.isChecked()
        
        if self.inference_active:
            self.inference_btn.setText("停止推理")
            self.status_bar.showMessage("模型推理已启用，正在处理当前帧...")
            if self.current_frame is not None:
                self.run_inference(self.current_frame)
                self.display_frame(self.current_frame)
        else:
            self.inference_btn.setText("开始推理")
            self.status_bar.showMessage("模型推理已禁用")
            self.inference_results = None
            # 刷新显示原始帧
            if self.current_frame is not None:
                self.display_frame(self.current_frame)
                
        self.update_button_styles()
        self.highlight_button(self.inference_btn)

    def lock_box(self, index):
        """锁定指定索引的框(1-based,对应数字键1-3)"""#0716
        if 1 <= index <= len(self.box_order):
            self.locked_box_index = index - 1 # 转换为0-based索引
            box = self.box_order[self.locked_box_index]['box']
            self.status_bar.showMessage(f"已锁定第{index}个框（从左到右）")
            return True
        else:
            self.status_bar.showMessage(f"没有第{index}个框，请检查框数量")
            return False    

    def load_model_dialog(self):
        """打开文件对话框加载检测模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择检测模型文件", 
            "", 
            "YOLO模型文件 (*.pt);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        # 加载检测模型
        try:
            self.detection_model = YOLO(file_path)
            self.status_bar.showMessage(f"检测模型已加载: {os.path.basename(file_path)}")
            self.highlight_button(self.load_model_btn)
            
            # 检查模型状态
            self.check_model_status()
        except Exception as e:
            QMessageBox.warning(self, "模型加载错误", f"无法加载检测模型: {str(e)}")
            self.model_loaded = False

    def manage_classification_models(self):
        """打开分类模型管理对话框"""
        dialog = ModelManagerDialog(self)
        dialog.set_models(self.classification_models)
        dialog.exec_()
    
    def add_classification_model(self, model, name):
        """添加一个新的分类模型"""
        self.classification_models.append({
            'model': model,
            'name': name
        })
        
        # 检查模型状态
        self.check_model_status()
        
        # 如果当前正在推理，重新运行推理
        if self.inference_active and self.current_frame is not None:
            self.run_inference(self.current_frame)
            self.display_frame(self.current_frame)
    
    def remove_classification_model(self, model_name):
        """删除指定的分类模型"""
        # 找到并删除模型
        for i, model_info in enumerate(self.classification_models):
            if model_info['name'] == model_name:
                del self.classification_models[i]
                break
        
        # 检查模型状态
        self.check_model_status()
        
        # 如果当前正在推理，重新运行推理
        if self.inference_active and self.current_frame is not None:
            self.run_inference(self.current_frame)
            self.display_frame(self.current_frame)
    
    def check_model_status(self):
        """检查模型状态并更新UI"""
        if self.detection_model is not None and self.classification_models:
            model_names = ", ".join([model['name'] for model in self.classification_models])
            self.model_loaded = True
            self.model_status_label.setText(f"模型已加载 ({model_names})")
            self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
            self.inference_btn.setEnabled(True)
        else:
            self.model_loaded = False
            if self.detection_model is None:
                self.model_status_label.setText("检测模型未加载")
            elif not self.classification_models:
                self.model_status_label.setText("分类模型未加载")
            else:
                self.model_status_label.setText("模型未加载")
                
            self.model_status_label.setStyleSheet("color: #FF5252; font-size: 12px;")
            self.inference_btn.setEnabled(False)
            
            # 如果推理按钮被按下，重置状态
            if self.inference_active:
                self.inference_active = False
                self.inference_btn.setChecked(False)
                self.inference_btn.setText("开始推理")

    def load_models(self):
        """加载默认检测模型（如果存在）"""
        # 检查默认模型文件是否存在
        detection_path = 'pot_414.pt'
        
        # 加载检测模型
        if os.path.exists(detection_path):
            try:
                self.detection_model = YOLO(detection_path)
                self.model_loaded = True
                self.model_status_label.setText("默认检测模型已加载")
                self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
            except Exception as e:
                print(f"无法加载检测模型: {str(e)}")
                self.model_loaded = False

    def run_inference(self, frame):
        """对当前帧运行模型推理"""
        if not self.model_loaded or frame is None:
            return None

        try:
            # 保存原始帧（用于后续保存操作）
            self.original_frame = frame.copy()  # 关键修改：保存原始帧
            
            # 进行目标检测
            detection_results = self.detection_model(frame)
            
            results = []
            for result in detection_results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 扩大检测框
                    scale_factor = 1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    width_patch = x2 - x1
                    height_patch = y2 - y1
                    
                    x1_new = int(center_x - width_patch * scale_factor / 2)
                    y1_new = int(center_y - height_patch * scale_factor / 2)
                    x2_new = int(center_x + width_patch * scale_factor / 2)
                    y2_new = int(center_y + height_patch * scale_factor / 2)
                    
                    # 确保坐标在图像范围内
                    x1_new = max(0, x1_new)
                    y1_new = max(0, y1_new)
                    x2_new = min(frame.shape[1] - 1, x2_new)
                    y2_new = min(frame.shape[0] - 1, y2_new)
                    
                    # DIS = 100
                    
                    # x1_new = x1_new + DIS
                    # y1_new = x2_new - DIS
                    # x2_new = y1_new + DIS  # 宽度边界
                    # y2_new = y2_new - DIS  # 高度边界

                    DIS = 0

                    # 提取边界框内的图像
                    cropped_image = frame[y1_new + DIS:y2_new - DIS, x1_new + DIS:x2_new - DIS]

                    # 提取并预处理图像
                    # ropped_image = frame[y1_new+200:y2_new-200, x1_new+150:x2_new-150]
                    [h, w, _] = cropped_image.shape
                    
                    if h == 0 or w == 0:
                        continue
                        
                    length = max((h, w))
                    image = np.zeros((length, length, 3), np.uint8)
                    image[0:h, 0:w] = cropped_image
                    
                    # 准备模型输入
                    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(224, 224), swapRB=True)
                    efficientnet_input = torch.from_numpy(blob)
                    
                    # 对每个分类模型进行推理
                    model_results = []
                    for model_info in self.classification_models:
                        model = model_info['model']
                        model_name = model_info['name']
                        
                        with torch.no_grad():
                            classification_results = model(efficientnet_input)
                            probabilities = torch.softmax(classification_results, dim=1)
                            max_prob, predicted_indices = torch.max(probabilities, dim=1)
                            
                            predicted_class_index = predicted_indices.item()
                            confidence = max_prob.item()
                            predicted_class = class_names.get(predicted_class_index, f"未知类别({predicted_class_index})")
                            
                            model_results.append({
                                'model_name': model_name,
                                'class_index': predicted_class_index,
                                'class_name': predicted_class,
                                'confidence': confidence
                            })
                    
                    results.append({
                        'box': (x1_new, y1_new, x2_new, y2_new),
                        'model_results': model_results
                    })
            
            # 对检测框按x1坐标从左到右排序 0716
            self.box_order = sorted(results, key=lambda x: x['box'][0])
            self.inference_results = self.box_order # 使用排序后的结果
            return self.box_order
            
        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            self.inference_results = None
            return None

    def draw_inference_results(self, frame):
        """在帧上绘制推理结果"""
        if self.inference_results is None:
            return frame
            
        for result in self.inference_results:
            box = result['box']
            x1, y1, x2, y2 = box
            
            # 为每个模型的结果绘制边界框和标签
            for i, model_result in enumerate(result['model_results']):
                class_name = model_result['class_name']
                confidence = model_result['confidence']
                model_name = model_result['model_name']
                color = class_colors.get(class_name, (255, 255, 255))
                
                # 只绘制一次边界框
                if i == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制模型名称和结果
                text = f"{model_name}: {class_name} ({confidence:.2f})"
                cv2.putText(frame, text, (x1, y1 + 30 * i + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    def save_current_frame(self, save_type=0):
        """保存当前帧到指定目录（保存原始帧，不包含可视化结果）"""
        if self.original_frame is None or not self.cap or not self.video_folder:
            return
        
        # 创建与视频文件夹同级的Handmade_images文件夹
        handmade_images_dir = os.path.join(os.path.dirname(self.video_folder), "Handmade_images")
        
        # 根据保存类型选择目录
        if save_type == 0:
            save_path = os.path.join(handmade_images_dir, "root")
            folder_name = "root文件夹"
        elif save_type == 1:
            save_path = os.path.join(handmade_images_dir, "0")
            folder_name = "0文件夹"
        elif save_type == 2:
            save_path = os.path.join(handmade_images_dir, "1")
            folder_name = "1文件夹"
        elif save_type == 3:
            save_path = os.path.join(handmade_images_dir, "2")
            folder_name = "2文件夹"
        else:  # save_type == 4
            save_path = os.path.join(handmade_images_dir, "3")
            folder_name = "3文件夹"

        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)

       # 原有代码：生成基础文件名
        video_name = os.path.splitext(self.video_files[self.current_video_index])[0]
        frame_num = self.current_frame_pos
        base_filename = f"{video_name}_frame_{frame_num:06d}"

        # 新增：如果锁定了框，保存框内图像  0716
        if self.locked_box_index is not None and 0 <= self.locked_box_index < len(self.box_order):
            # 提取锁定框的坐标
            x1, y1, x2, y2 = self.box_order[self.locked_box_index]['box']
            # 裁剪框内图像（使用原始帧）
            box_image = self.original_frame[y1:y2, x1:x2]
            if box_image.size > 0:  # 确保裁剪有效
                filename = os.path.join(save_path, f"{base_filename}_box{self.locked_box_index+1}.jpg")
                cv2.imwrite(filename, box_image)
                self.last_saved_file = filename  # 记录最后保存的文件路径
                self.status_bar.showMessage(f"已保存锁定框图像到{folder_name}：{os.path.basename(filename)}")
                return

        # 原有逻辑：未锁定框时保存整帧
        filename = os.path.join(save_path, f"{base_filename}.jpg")
        cv2.imwrite(filename, self.original_frame)
        self.last_saved_file = filename  # 记录最后保存的文件路径
        self.status_bar.showMessage(f"已保存原始帧到{folder_name}：{os.path.basename(filename)}")

    def undo_last_save(self):
        """撤回上一张保存的图片，移动到mistake文件夹"""
        if self.last_saved_file is None or not os.path.exists(self.last_saved_file):
            self.status_bar.showMessage("没有可撤回的文件")
            return
        
        try:
            # 创建mistake文件夹
            handmade_images_dir = os.path.join(os.path.dirname(self.video_folder), "Handmade_images")
            mistake_dir = os.path.join(handmade_images_dir, "mistake")
            os.makedirs(mistake_dir, exist_ok=True)
            
            # 移动文件到mistake文件夹
            filename = os.path.basename(self.last_saved_file)
            mistake_path = os.path.join(mistake_dir, filename)
            
            # 如果目标文件已存在，添加时间戳避免覆盖
            if os.path.exists(mistake_path):
                import time
                timestamp = int(time.time())
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{timestamp}{ext}"
                mistake_path = os.path.join(mistake_dir, filename)
            
            # 移动文件
            import shutil
            shutil.move(self.last_saved_file, mistake_path)
            
            self.status_bar.showMessage(f"已撤回文件到mistake文件夹：{filename}")
            self.last_saved_file = None  # 清除记录
            
        except Exception as e:
            self.status_bar.showMessage(f"撤回失败：{str(e)}")

    # 按钮高亮方法
    def highlight_button(self, button):
        """高亮显示指定的按钮"""
        # 清除之前的高亮
        self.clear_button_highlight()
        
        # 设置新的高亮按钮
        self.active_button = button
        self.update_button_styles()
        
        # 设置定时器清除高亮
        self.button_highlight_timer.start(self.button_highlight_duration)

    def clear_button_highlight(self):
        """清除按钮高亮状态"""
        self.active_button = None
        self.button_highlight_timer.stop()
        self.update_button_styles()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
