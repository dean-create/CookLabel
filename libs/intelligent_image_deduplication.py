#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能图片去重工具 - 基于深度学习特征提取的图片去重
支持用户图形化选择输入文件夹、输出文件夹，并自定义重复检测阈值
"""

import os
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import time
from datetime import timedelta
import glob
from sklearn.preprocessing import normalize
import cv2

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    pyqt_version = 5
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    pyqt_version = 4

try:
    from libs.utils import new_icon
except ImportError:
    from utils import new_icon


class IntelligentImageDeduplicationDialog(QDialog):
    """智能图片去重工具对话框"""
    
    # 定义信号，用于线程间通信
    if pyqt_version == 5:
        progress_updated = pyqtSignal(int)  # 进度更新信号
        log_updated = pyqtSignal(str)      # 日志更新信号
        processing_finished = pyqtSignal(dict)  # 处理完成信号
        stage_updated = pyqtSignal(str)    # 阶段更新信号
    else:
        progress_updated = pyqtSignal(int)  # 进度更新信号
        log_updated = pyqtSignal(str)      # 日志更新信号
        processing_finished = pyqtSignal(dict)  # 处理完成信号
        stage_updated = pyqtSignal(str)    # 阶段更新信号
    
    def __init__(self, parent=None):
        super(IntelligentImageDeduplicationDialog, self).__init__(parent)
        self.setWindowTitle("智能图片去重工具")
        self.setWindowIcon(new_icon('app'))
        self.setMinimumSize(1000, 750)  # 设置最小尺寸
        self.resize(1200, 850)          # 设置默认尺寸
        
        # 初始化变量
        self.input_folder_path = ""
        self.output_folder_path = ""
        self.is_processing = False      # 是否正在处理
        self.processing_thread = None   # 处理线程
        
        # 重复检测阈值
        self.hamming_threshold = 0.85   # 汉明相似度阈值
        self.cosine_threshold = 0.850   # 余弦相似度阈值
        
        # 初始化界面
        self.init_ui()
        
        # 连接信号
        self.progress_updated.connect(self.update_progress)
        self.log_updated.connect(self.append_log)
        self.processing_finished.connect(self.on_processing_finished)
        self.stage_updated.connect(self.update_stage)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("智能图片去重工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; margin: 15px; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # 功能说明
        description_label = QLabel("基于深度学习特征提取，智能检测并去除重复图片，便于人工检查")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("font-size: 20px; color: #7f8c8d; margin-bottom: 10px;")
        layout.addWidget(description_label)
        
        # 路径选择区域
        path_group = QGroupBox("路径设置")
        path_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        path_layout = QVBoxLayout()
        
        # 输入文件夹选择
        input_folder_label_title = QLabel("输入图片文件夹:")
        input_folder_label_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        path_layout.addWidget(input_folder_label_title)
        
        input_folder_layout = QHBoxLayout()
        self.input_folder_label = QLabel("请选择包含图片的文件夹...")
        self.input_folder_label.setStyleSheet("""
            border: 1px solid #bdc3c7; 
            padding: 10px; 
            background-color: #ecf0f1;
            border-radius: 5px;
            min-height: 25px;
        """)
        self.input_folder_label.setWordWrap(True)  # 允许文本换行
        self.input_folder_btn = QPushButton("浏览")
        self.input_folder_btn.setFixedWidth(80)  # 固定按钮宽度
        self.input_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.input_folder_btn.clicked.connect(self.select_input_folder)
        input_folder_layout.addWidget(self.input_folder_label, 1)
        input_folder_layout.addWidget(self.input_folder_btn)
        path_layout.addLayout(input_folder_layout)
        
        # 添加间距
        path_layout.addSpacing(15)
        
        # 输出文件夹选择
        output_folder_label_title = QLabel("输出文件夹:")
        output_folder_label_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        path_layout.addWidget(output_folder_label_title)
        
        output_folder_layout = QHBoxLayout()
        self.output_folder_label = QLabel("请选择输出文件夹...")
        self.output_folder_label.setStyleSheet("""
            border: 1px solid #bdc3c7; 
            padding: 10px; 
            background-color: #ecf0f1;
            border-radius: 5px;
            min-height: 25px;
        """)
        self.output_folder_label.setWordWrap(True)  # 允许文本换行
        self.output_folder_btn = QPushButton("浏览")
        self.output_folder_btn.setFixedWidth(80)  # 固定按钮宽度
        self.output_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.output_folder_btn.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(self.output_folder_label, 1)
        output_folder_layout.addWidget(self.output_folder_btn)
        path_layout.addLayout(output_folder_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 重复检测阈值设置
        threshold_group = QGroupBox("重复检测阈值设置")
        threshold_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(15)
        
        # 汉明相似度阈值设置
        hamming_layout = QVBoxLayout()
        hamming_title = QLabel("汉明相似度阈值 (pHash初筛):")
        hamming_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        hamming_layout.addWidget(hamming_title)
        
        hamming_control_layout = QHBoxLayout()
        self.hamming_slider = QSlider(Qt.Horizontal)
        self.hamming_slider.setRange(0, 100)  # 0.00 到 1.00，步长0.01
        self.hamming_slider.setValue(85)      # 默认0.85
        self.hamming_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 4px;
            }
        """)
        self.hamming_value_label = QLabel("0.85")
        self.hamming_value_label.setStyleSheet("font-weight: bold; color: #2c3e50; min-width: 40px;")
        self.hamming_slider.valueChanged.connect(self.update_hamming_threshold)
        
        hamming_control_layout.addWidget(QLabel("0.00"))
        hamming_control_layout.addWidget(self.hamming_slider)
        hamming_control_layout.addWidget(QLabel("1.00"))
        hamming_control_layout.addWidget(self.hamming_value_label)
        hamming_layout.addLayout(hamming_control_layout)
        
        hamming_desc = QLabel("值越大越严格，推荐0.80-0.90")
        hamming_desc.setStyleSheet("color: #7f8c8d; font-size: 18px;")
        hamming_layout.addWidget(hamming_desc)
        threshold_layout.addLayout(hamming_layout)
        
        # 余弦相似度阈值设置
        cosine_layout = QVBoxLayout()
        cosine_title = QLabel("余弦相似度阈值 (ResNet50精筛):")
        cosine_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        cosine_layout.addWidget(cosine_title)
        
        cosine_control_layout = QHBoxLayout()
        self.cosine_slider = QSlider(Qt.Horizontal)
        self.cosine_slider.setRange(0, 1000)  # 0.000 到 1.000，步长0.001
        self.cosine_slider.setValue(850)      # 默认0.850
        self.cosine_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #27ae60;
                border: 1px solid #229954;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #27ae60;
                border-radius: 4px;
            }
        """)
        self.cosine_value_label = QLabel("0.850")
        self.cosine_value_label.setStyleSheet("font-weight: bold; color: #2c3e50; min-width: 40px;")
        self.cosine_slider.valueChanged.connect(self.update_cosine_threshold)
        
        cosine_control_layout.addWidget(QLabel("0.000"))
        cosine_control_layout.addWidget(self.cosine_slider)
        cosine_control_layout.addWidget(QLabel("1.000"))
        cosine_control_layout.addWidget(self.cosine_value_label)
        cosine_layout.addLayout(cosine_control_layout)
        
        cosine_desc = QLabel("值越大越严格，推荐0.800-0.900")
        cosine_desc.setStyleSheet("color: #7f8c8d; font-size: 18px;")
        cosine_layout.addWidget(cosine_desc)
        threshold_layout.addLayout(cosine_layout)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # 进度显示区域
        progress_group = QGroupBox("处理进度")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        
        # 当前阶段显示
        self.stage_label = QLabel("等待开始...")
        self.stage_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 14px;")
        progress_layout.addWidget(self.stage_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
                padding: 10px;
                min-height: 25px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setFixedHeight(45)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.setFixedHeight(45)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        # 添加弹性空间使按钮居中
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 日志显示区域
        log_group = QGroupBox("处理日志")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                background-color: #ffffff;
                color: #000000;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
                border: 1px solid #d0d0d0;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 5px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #808080;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    def select_input_folder(self):
        """选择输入文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输入图片文件夹")
        if folder:
            self.input_folder_path = folder
            self.input_folder_label.setText(folder)
            self.append_log(f"已选择输入文件夹: {folder}")
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_path = folder
            self.output_folder_label.setText(folder)
            self.append_log(f"已选择输出文件夹: {folder}")
    

    
    def update_hamming_threshold(self, value):
        """更新汉明相似度阈值"""
        self.hamming_threshold = value / 100.0  # 转换为0.00-1.00范围
        self.hamming_value_label.setText(f"{self.hamming_threshold:.2f}")
    
    def update_cosine_threshold(self, value):
        """更新余弦相似度阈值"""
        self.cosine_threshold = value / 1000.0  # 转换为0.000-1.000范围
        self.cosine_value_label.setText(f"{self.cosine_threshold:.3f}")
    
    def start_processing(self):
        """开始处理"""
        # 验证输入
        if not self.input_folder_path:
            QMessageBox.warning(self, "警告", "请先选择输入文件夹！")
            return
        
        if not self.output_folder_path:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return
        
        if not os.path.exists(self.input_folder_path):
            QMessageBox.warning(self, "警告", "输入文件夹不存在！")
            return
        
        # 更新界面状态
        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("处理中...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        
        # 清空日志
        self.log_text.clear()
        self.append_log("开始智能图片去重处理...")
        
        # 创建并启动处理线程
        self.processing_thread = ImageProcessingThread(
            self.input_folder_path,
            self.output_folder_path,
            self.hamming_threshold,
            self.cosine_threshold,
            self
        )
        self.processing_thread.start()
    
    def stop_processing(self):
        """停止处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.append_log("正在停止处理...")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_stage(self, stage):
        """更新当前阶段"""
        self.stage_label.setText(stage)
    
    def append_log(self, message):
        """添加日志信息"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_processing_finished(self, results):
        """处理完成"""
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if results.get('success', False):
            self.progress_bar.setValue(100)
            self.status_label.setText("处理完成")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            # 显示统计信息
            stats = results.get('stats', {})
            self.append_log("=" * 50)
            self.append_log("智能图片去重处理完成！")
            self.append_log(f"处理总耗时: {stats.get('elapsed_time', '未知')}")
            self.append_log(f"输入图片数量: {stats.get('input_count', 0)}")
            self.append_log(f"检测到重复图片: {stats.get('duplicate_count', 0)} 张")
            self.append_log(f"去重后唯一图片: {stats.get('unique_count', 0)} 张")
            self.append_log(f"输出去重图片数量: {stats.get('output_count', 0)}")
            self.append_log(f"输出重复图片数量: {stats.get('duplicate_output_count', 0)}")
            duplicate_rate = stats.get('duplicate_count', 0) / max(stats.get('input_count', 1), 1) * 100
            self.append_log(f"去重率: {duplicate_rate:.2f}%")
            self.append_log("=" * 50)
            
            # 显示完成对话框
            QMessageBox.information(
                self, 
                "完成",
 
                f"智能图片去重处理完成！\n\n"
                f"输入图片数量: {stats.get('input_count', 0)}\n"
                f"检测到重复图片: {stats.get('duplicate_count', 0)} 张\n"
                f"去重后唯一图片: {stats.get('unique_count', 0)} 张\n"
                f"输出去重图片数量: {stats.get('output_count', 0)}\n"
                f"输出重复图片数量: {stats.get('duplicate_output_count', 0)}\n"
                f"去重率: {duplicate_rate:.2f}%\n"
                f"处理耗时: {stats.get('elapsed_time', '未知')}"
            )
        else:
            self.status_label.setText("处理失败或被中断")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            error_msg = results.get('error', '未知错误')
            self.append_log(f"处理失败: {error_msg}")
            QMessageBox.warning(self, "错误", f"智能图片去重处理失败:\n{error_msg}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.is_processing:
            reply = QMessageBox.question(
                self, 
                "确认关闭", 
                "图片处理正在进行中，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class ImageProcessingThread(QThread):
    """图片处理线程"""
    
    def __init__(self, input_folder, output_folder, hamming_threshold, cosine_threshold, parent_dialog):
        super(ImageProcessingThread, self).__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.hamming_threshold = hamming_threshold
        self.cosine_threshold = cosine_threshold
        self.parent_dialog = parent_dialog
        self.should_stop = False
        
        # 初始化深度学习模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.parent_dialog.log_updated.emit(f"使用设备: {self.device}")
        
    def stop(self):
        """停止线程"""
        self.should_stop = True
    
    def run(self):
        """线程主函数"""
        try:
            start_time = time.time()
            
            # 第一阶段：初始化模型
            self.parent_dialog.stage_updated.emit("阶段 1/5: 初始化深度学习模型...")
            self.parent_dialog.progress_updated.emit(5)
            
            # 加载预训练的ResNet50模型用于提取图像特征
            # 使用新版本的weights参数替代过时的pretrained参数，消除警告信息
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # 移除最后的全连接层，只使用特征提取部分
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_extractor.to(self.device)
            feature_extractor.eval()
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.parent_dialog.log_updated.emit("深度学习模型初始化完成")
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 第二阶段：扫描图片文件
            self.parent_dialog.stage_updated.emit("阶段 2/5: 扫描图片文件...")
            self.parent_dialog.progress_updated.emit(10)
            
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            image_paths = []
            
            for root, _, files in os.walk(self.input_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(root, file))
            
            self.parent_dialog.log_updated.emit(f"发现 {len(image_paths)} 张图片")
            
            if len(image_paths) == 0:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '输入文件夹中没有找到图片文件'})
                return
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 第三阶段：提取图像特征
            self.parent_dialog.stage_updated.emit("阶段 3/5: 提取图像特征...")
            self.parent_dialog.progress_updated.emit(20)
            
            features = []
            valid_paths = []
            
            for i, path in enumerate(image_paths):
                if self.should_stop:
                    self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                    return
                
                feat = self.extract_features(path, feature_extractor, transform)
                if feat is not None:
                    features.append(feat)
                    valid_paths.append(path)
                
                # 更新进度
                progress = 20 + int((i / len(image_paths)) * 30)
                self.parent_dialog.progress_updated.emit(progress)
                
                if (i + 1) % 100 == 0:
                    self.parent_dialog.log_updated.emit(f"已提取 {i + 1}/{len(image_paths)} 张图片的特征")
            
            if len(features) == 0:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '没有成功提取到任何图片特征'})
                return
            
            features = np.array(features)
            self.parent_dialog.log_updated.emit(f"成功提取 {len(features)} 张图片的2048维特征")
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 第四阶段：两阶段重复检测
            self.parent_dialog.stage_updated.emit("阶段 4/5: 两阶段重复检测（pHash + ResNet50）...")
            self.parent_dialog.progress_updated.emit(50)
            
            # 对所有图片进行两阶段重复检测
            duplicate_pairs, unique_indices, stage_stats = self.detect_duplicates_two_stage(features, valid_paths)
            
            # 统计重复检测结果
            total_input_count = len(valid_paths)
            total_duplicate_count = total_input_count - len(unique_indices)
            unique_paths = [valid_paths[i] for i in unique_indices]
            
            self.parent_dialog.log_updated.emit(f"重复检测完成：")
            self.parent_dialog.log_updated.emit(f"  - 输入图片总数：{total_input_count}")
            self.parent_dialog.log_updated.emit(f"  - 检测到重复图片：{total_duplicate_count} 张")
            self.parent_dialog.log_updated.emit(f"  - 去重后唯一图片：{len(unique_paths)} 张")
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 第五阶段：保存去重后的图片和重复图片
            self.parent_dialog.stage_updated.emit("阶段 5/5: 保存去重后的图片和重复图片...")
            self.parent_dialog.progress_updated.emit(80)
            
            # 创建输出目录
            deduplicated_folder = os.path.join(self.output_folder, "deduplicated_images")
            duplicated_folder = os.path.join(self.output_folder, "duplicated_images")
            os.makedirs(deduplicated_folder, exist_ok=True)
            os.makedirs(duplicated_folder, exist_ok=True)
            
            total_output_count = 0
            total_duplicate_output_count = 0
            
            # 复制去重后的图片
            for i, src_path in enumerate(unique_paths):
                if self.should_stop:
                    self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                    return
                
                filename = os.path.basename(src_path)
                name, ext = os.path.splitext(filename)
                
                # 保持原始文件名，处理重名冲突
                dest_path = os.path.join(deduplicated_folder, filename)
                counter = 1
                while os.path.exists(dest_path):
                    name_part, ext_part = os.path.splitext(filename)
                    dest_path = os.path.join(deduplicated_folder, f"{name_part}({counter}){ext_part}")
                    counter += 1
                
                try:
                    shutil.copy2(src_path, dest_path)
                    total_output_count += 1
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"复制去重图片失败: {e}")
                
                # 更新进度（前70%用于去重图片）
                progress = 80 + int((i + 1) / len(unique_paths) * 10)
                self.parent_dialog.progress_updated.emit(progress)
                
                if (i + 1) % 50 == 0:
                    self.parent_dialog.log_updated.emit(f"已保存 {i + 1}/{len(unique_paths)} 张去重后的图片")
            
            # 提取重复图片的索引（从duplicate_pairs中获取被标记为重复的图片）
            duplicate_indices = set()
            for idx1, idx2, cosine_sim, hamming_sim in duplicate_pairs:
                # idx2是被标记为重复的图片索引
                duplicate_indices.add(idx2)
            
            # 复制重复的图片到duplicated_images文件夹
            duplicate_paths = [valid_paths[idx] for idx in duplicate_indices]
            
            self.parent_dialog.log_updated.emit(f"开始保存 {len(duplicate_paths)} 张重复图片...")
            
            for i, src_path in enumerate(duplicate_paths):
                if self.should_stop:
                    self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                    return
                
                filename = os.path.basename(src_path)
                name, ext = os.path.splitext(filename)
                
                # 保持原始文件名，处理重名冲突
                dest_path = os.path.join(duplicated_folder, filename)
                counter = 1
                while os.path.exists(dest_path):
                    name_part, ext_part = os.path.splitext(filename)
                    dest_path = os.path.join(duplicated_folder, f"{name_part}({counter}){ext_part}")
                    counter += 1
                
                try:
                    shutil.copy2(src_path, dest_path)
                    total_duplicate_output_count += 1
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"复制重复图片失败: {e}")
                
                # 更新进度（后20%用于重复图片）
                if len(duplicate_paths) > 0:
                    progress = 90 + int((i + 1) / len(duplicate_paths) * 5)
                    self.parent_dialog.progress_updated.emit(progress)
                
                if (i + 1) % 50 == 0:
                    self.parent_dialog.log_updated.emit(f"已保存 {i + 1}/{len(duplicate_paths)} 张重复图片")
            
            self.parent_dialog.progress_updated.emit(95)
            
            # 生成处理报告
            report = self.generate_report(total_duplicate_count, unique_paths, deduplicated_folder, stage_stats, duplicated_folder, total_duplicate_output_count)
            
            # 计算统计信息
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))
            
            self.parent_dialog.log_updated.emit("=" * 60)
            self.parent_dialog.log_updated.emit("智能图片去重处理完成！")
            self.parent_dialog.log_updated.emit(f"输入图片数量: {total_input_count}")
            self.parent_dialog.log_updated.emit(f"检测到重复图片: {total_duplicate_count} 张")
            self.parent_dialog.log_updated.emit(f"去重后唯一图片: {len(unique_paths)} 张")
            self.parent_dialog.log_updated.emit(f"输出去重图片数量: {total_output_count}")
            self.parent_dialog.log_updated.emit(f"输出重复图片数量: {total_duplicate_output_count}")
            self.parent_dialog.log_updated.emit(f"处理耗时: {elapsed_formatted}")
            self.parent_dialog.log_updated.emit(f"去重图片输出目录: {deduplicated_folder}")
            self.parent_dialog.log_updated.emit(f"重复图片输出目录: {duplicated_folder}")
            self.parent_dialog.log_updated.emit("=" * 60)
            
            self.parent_dialog.progress_updated.emit(100)
            
            # 发送完成信号
            self.parent_dialog.processing_finished.emit({
                'success': True,
                'total_images': total_input_count,
                'duplicate_count': total_duplicate_count,
                'unique_count': len(unique_paths),
                'deduplicated_folder': deduplicated_folder,
                'duplicated_folder': duplicated_folder,
                'report': report,
                'stats': {
                    'elapsed_time': elapsed_formatted,
                    'input_count': total_input_count,
                    'duplicate_count': total_duplicate_count,
                    'unique_count': len(unique_paths),
                    'output_count': total_output_count,
                    'duplicate_output_count': total_duplicate_output_count
                }
            })
            
        except Exception as e:
            self.parent_dialog.processing_finished.emit({
                'success': False,
                'error': str(e)
            })
    
    def extract_features(self, image_path, feature_extractor, transform):
        """提取单张图像的特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = feature_extractor(image)
                feature = feature.squeeze().cpu().numpy()
            
            # L2归一化特征向量，为余弦相似度计算做准备
            return normalize(feature.reshape(1, -1))[0]
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"处理图像 {image_path} 时出错: {str(e)}")
            return None
    
    def calculate_perceptual_hash(self, image_path, hash_size=8):
        """
        计算图像的感知哈希值
        
        Args:
            image_path: 图像文件路径
            hash_size: 哈希大小，默认8x8=64位
            
        Returns:
            hash_value: 感知哈希值（整数）
        """
        try:
            # 打开图像并转换为灰度
            image = Image.open(image_path).convert('L')
            
            # 缩放到hash_size x hash_size
            image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            
            # 转换为numpy数组
            pixels = np.array(image)
            
            # 计算平均值
            avg = pixels.mean()
            
            # 生成哈希：像素值大于平均值为1，否则为0
            hash_bits = (pixels > avg).flatten()
            
            # 将二进制数组转换为整数
            hash_value = 0
            for i, bit in enumerate(hash_bits):
                if bit:
                    hash_value |= (1 << i)
            
            return hash_value
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"计算感知哈希失败 {image_path}: {str(e)}")
            return None
    
    def hamming_distance(self, hash1, hash2):
        """
        计算两个哈希值的汉明距离
        
        Args:
            hash1, hash2: 感知哈希值
            
        Returns:
            distance: 汉明距离（不同位的数量）
        """
        return bin(hash1 ^ hash2).count('1')
    
    def hamming_similarity(self, hash1, hash2, hash_length=64):
        """
        计算两个哈希值的汉明相似度分数
        
        Args:
            hash1, hash2: 感知哈希值
            hash_length: 哈希长度（位数），默认64位
            
        Returns:
            similarity: 汉明相似度分数（0-1之间，值越大越相似）
        """
        # 计算汉明距离
        hamming_bits = self.hamming_distance(hash1, hash2)
        # 归一化汉明距离
        normalized_distance = hamming_bits / hash_length
        # 转换为相似度分数：1 - 归一化距离
        similarity = 1.0 - normalized_distance
        return similarity
    
    def detect_duplicates_two_stage(self, features, image_paths):
        """
        使用两阶段筛选策略检测重复图片：
        第一阶段：pHash快速初筛（汉明相似度）
        第二阶段：ResNet50精确筛选（余弦相似度）
        
        Args:
            features: L2归一化的特征向量数组
            image_paths: 图片路径列表
            
        Returns:
            duplicate_pairs: 重复图片对列表 [(idx1, idx2, cosine_sim, hamming_sim)]
            unique_indices: 去重后的唯一图片索引列表
            stage_stats: 两阶段统计信息 {'stage1_candidates': int, 'stage2_duplicates': int}
        """
        try:
            self.parent_dialog.log_updated.emit("开始两阶段重复检测...")
            self.parent_dialog.log_updated.emit(f"阈值设置：汉明相似度 ≥ {self.hamming_threshold:.2f}，余弦相似度 ≥ {self.cosine_threshold:.3f}")
            
            # 第一阶段：计算所有图片的感知哈希（pHash快速初筛）
            self.parent_dialog.log_updated.emit("第一阶段：计算感知哈希进行快速初筛...")
            
            perceptual_hashes = []
            hash_length = 64  # 8x8 = 64位哈希
            
            for i, path in enumerate(image_paths):
                if self.should_stop:
                    stage_stats = {'stage1_candidates': 0, 'stage2_duplicates': 0}
                    return [], list(range(len(image_paths))), stage_stats
                
                hash_val = self.calculate_perceptual_hash(path)
                perceptual_hashes.append(hash_val)
                
                if (i + 1) % 50 == 0:
                    self.parent_dialog.log_updated.emit(f"已计算 {i + 1}/{len(image_paths)} 张图片的感知哈希")
            
            # 第一阶段筛选：基于汉明相似度的快速初筛
            candidate_pairs = []
            total_comparisons = 0
            
            for i in range(len(perceptual_hashes)):
                if self.should_stop:
                    stage_stats = {'stage1_candidates': 0, 'stage2_duplicates': 0}
                    return [], list(range(len(image_paths))), stage_stats
                    
                for j in range(i + 1, len(perceptual_hashes)):
                    total_comparisons += 1
                    
                    # 计算汉明相似度
                    if perceptual_hashes[i] is not None and perceptual_hashes[j] is not None:
                        hamming_sim = self.hamming_similarity(perceptual_hashes[i], perceptual_hashes[j], hash_length)
                        
                        # 第一阶段筛选：汉明相似度阈值（值越大越相似）
                        if hamming_sim >= self.hamming_threshold:
                            candidate_pairs.append((i, j, hamming_sim))
            
            self.parent_dialog.log_updated.emit(f"第一阶段完成：从 {total_comparisons} 对比较中筛选出 {len(candidate_pairs)} 对候选重复图片")
            
            if len(candidate_pairs) == 0:
                self.parent_dialog.log_updated.emit("第一阶段未发现候选重复图片，无需进行第二阶段筛选")
                stage_stats = {'stage1_candidates': 0, 'stage2_duplicates': 0}
                return [], list(range(len(image_paths))), stage_stats
            
            # 第二阶段：对候选图片对进行ResNet50特征的精确筛选
            self.parent_dialog.log_updated.emit("第二阶段：使用ResNet50特征进行精确筛选...")
            
            # 计算余弦相似度矩阵（特征已L2归一化，直接用点积）
            similarity_matrix = np.dot(features, features.T)
            
            # 第二阶段筛选：基于余弦相似度的精确筛选
            duplicate_pairs = []
            processed_indices = set()
            
            for i, j, hamming_sim in candidate_pairs:
                if i in processed_indices or j in processed_indices:
                    continue
                
                # 计算余弦相似度
                cosine_sim = similarity_matrix[i, j]
                
                # 第二阶段筛选：余弦相似度阈值
                if cosine_sim >= self.cosine_threshold:
                    duplicate_pairs.append((i, j, cosine_sim, hamming_sim))
                    processed_indices.add(j)  # 标记j为重复，保留i
            
            # 生成唯一图片索引列表
            unique_indices = [i for i in range(len(features)) if i not in processed_indices]
            
            self.parent_dialog.log_updated.emit(f"第二阶段完成：最终确认 {len(duplicate_pairs)} 对重复图片")
            
            # 输出检测结果统计
            if duplicate_pairs:
                self.parent_dialog.log_updated.emit("两阶段筛选统计：")
                self.parent_dialog.log_updated.emit(f"  - 总图片数：{len(image_paths)}")
                self.parent_dialog.log_updated.emit(f"  - 第一阶段候选对：{len(candidate_pairs)}")
                self.parent_dialog.log_updated.emit(f"  - 第二阶段确认重复对：{len(duplicate_pairs)}")
                self.parent_dialog.log_updated.emit(f"  - 重复图片数：{len(processed_indices)}")
                self.parent_dialog.log_updated.emit(f"  - 唯一图片数：{len(unique_indices)}")
                
                # 输出前几对重复图片的详细信息
                self.parent_dialog.log_updated.emit("检测到的重复图片示例（前5对）：")
                for idx, (i, j, cos_sim, hamming_sim) in enumerate(duplicate_pairs[:5]):
                    img1_name = os.path.basename(image_paths[i])
                    img2_name = os.path.basename(image_paths[j])
                    self.parent_dialog.log_updated.emit(
                        f"  {idx+1}. {img1_name} ↔ {img2_name} "
                        f"(余弦相似度: {cos_sim:.4f}, 汉明相似度: {hamming_sim:.3f})"
                    )
            
            # 返回结果和统计信息
            stage_stats = {
                'stage1_candidates': len(candidate_pairs),
                'stage2_duplicates': len(duplicate_pairs)
            }
            return duplicate_pairs, unique_indices, stage_stats
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"两阶段重复检测过程出错: {str(e)}")
            # 出错时返回所有图片作为唯一图片
            stage_stats = {'stage1_candidates': 0, 'stage2_duplicates': 0}
            return [], list(range(len(features))), stage_stats
    

    

    
    def max_min_sampling(self, features, num_samples):
        """
        使用最大最小距离算法选择多样化的样本
        1. 先选择距离最远的两个样本
        2. 每次选择距离已选样本集最远的样本
        """
        num_features = features.shape[0]
        if num_samples >= num_features:
            return list(range(num_features))
        
        if num_samples <= 0:
            return []
        
        if num_samples == 1:
            return [0]
        
        # 计算所有特征之间的距离矩阵
        distances = np.zeros((num_features, num_features))
        for i in range(num_features):
            for j in range(i+1, num_features):
                dist = np.linalg.norm(features[i] - features[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 找到距离最远的两个样本
        max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
        selected = list(max_dist_idx)
        
        # 迭代选择剩余样本
        for _ in range(num_samples - 2):
            min_distances = []
            for i in range(num_features):
                if i not in selected:
                    # 计算到已选样本的最小距离
                    min_dist = np.min([distances[i, j] for j in selected])
                    min_distances.append((i, min_dist))
            
            if not min_distances:
                break
            
            # 选择最小距离最大的样本
            min_distances.sort(key=lambda x: x[1], reverse=True)
            selected.append(min_distances[0][0])
        
        return selected
    
    def generate_report(self, duplicate_count, unique_paths, output_folder, stage_stats, duplicated_folder=None, duplicate_output_count=0):
        """生成处理报告"""
        try:
            report_path = os.path.join(output_folder, "deduplication_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("智能图片去重处理报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入图片数量: {len(unique_paths) + duplicate_count}\n")
                f.write(f"检测到重复图片: {duplicate_count} 张\n")
                f.write(f"去重后唯一图片: {len(unique_paths)} 张\n")
                f.write(f"输出去重图片数量: {len(unique_paths)}\n")
                f.write(f"输出重复图片数量: {duplicate_output_count}\n")
                f.write(f"去重率: {duplicate_count / (len(unique_paths) + duplicate_count) * 100:.2f}%\n\n")
                
                f.write("处理算法详细信息:\n")
                f.write("-" * 40 + "\n")
                f.write("两阶段重复检测策略:\n")
                f.write(f"1. 第一阶段（pHash快速初筛）:\n")
                f.write(f"   - 使用感知哈希算法计算64位图像指纹\n")
                f.write(f"   - 归一化汉明相似度阈值: ≥ {self.hamming_threshold:.2f}\n")
                f.write(f"   - 快速筛选候选重复图片对\n")
                f.write(f"   - 初筛筛出候选对数量: {stage_stats.get('stage1_candidates', 0)} 对\n\n")
                f.write(f"2. 第二阶段（ResNet50精确筛选）:\n")
                f.write(f"   - 使用深度学习ResNet50模型提取2048维图像特征\n")
                f.write(f"   - L2归一化后计算余弦相似度\n")
                f.write(f"   - 余弦相似度阈值: ≥ {self.cosine_threshold:.3f}\n")
                f.write(f"   - 精确确认重复图片\n")
                f.write(f"   - 精筛确认重复对数量: {stage_stats.get('stage2_duplicates', 0)} 对\n\n")
                
                f.write("算法优势:\n")
                f.write("- 像素/结构级别检测：pHash算法检测图像结构相似性\n")
                f.write("- 语义级别检测：ResNet50深度特征检测语义相似性\n")
                f.write("- 高效性：两阶段筛选大幅减少计算量\n")
                f.write("- 准确性：双重验证机制确保检测精度\n")
                f.write("- 可定制：用户可自定义检测阈值\n\n")
                
                f.write("输出说明:\n")
                f.write("- deduplicated_images文件夹包含所有去重后的唯一图片\n")
                f.write("- duplicated_images文件夹包含所有检测到的重复图片\n")
                f.write("- 去重图片文件名格式：原名_序号.扩展名\n")
                f.write("- 重复图片文件名格式：原名_dup_序号.扩展名\n")
                f.write("- 建议人工抽查结果，确认去重效果\n")
                f.write("- 可通过对比两个文件夹验证重复检测的准确性\n")
                
                f.write("=" * 50 + "\n")
            
            self.parent_dialog.log_updated.emit(f"处理报告已保存到: {report_path}")
            return report_path
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"生成报告时出错: {e}")
            return None


def show_intelligent_image_deduplication(parent=None):
    """显示智能图片去重工具对话框"""
    dialog = IntelligentImageDeduplicationDialog(parent)
    return dialog.exec_()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    dialog = IntelligentImageDeduplicationDialog()
    dialog.show()
    
    sys.exit(app.exec_())