#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像去模糊检测工具 - 基于Laplace方差法的模糊图片检测
支持用户图形化选择输入文件夹、输出文件夹，并自定义模糊检测阈值
"""

import os
import shutil
import numpy as np
import cv2
from PIL import Image
import time
from datetime import timedelta
import glob

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


class ImageDeblurDetectionDialog(QDialog):
    """图像去模糊检测工具对话框"""
    
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
        super(ImageDeblurDetectionDialog, self).__init__(parent)
        self.setWindowTitle("图像去模糊检测工具")
        self.setWindowIcon(new_icon('app'))
        self.setMinimumSize(1000, 750)  # 设置最小尺寸
        self.resize(1200, 850)          # 设置默认尺寸
        
        # 初始化变量
        self.input_folder_path = ""
        self.output_folder_path = ""
        self.is_processing = False      # 是否正在处理
        self.processing_thread = None   # 处理线程
        
        # 模糊检测阈值（Laplace方差法）
        self.blur_threshold = 100.0     # 方差阈值，小于此值认为是模糊图片
        
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
        title_label = QLabel("图像去模糊检测工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; margin: 15px; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # 功能说明
        description_label = QLabel("基于Laplace方差法检测模糊图片，自动筛选清晰图像")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("font-size: 18px; color: #7f8c8d; margin-bottom: 10px;")
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
        
        # 模糊检测阈值设置
        threshold_group = QGroupBox("模糊检测阈值设置")
        threshold_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(15)
        
        # Laplace方差阈值设置
        blur_layout = QVBoxLayout()
        blur_title = QLabel("Laplace方差阈值 (小于此值认为模糊):")
        blur_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        blur_layout.addWidget(blur_title)
        
        blur_control_layout = QHBoxLayout()
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(10, 500)  # 10 到 500
        self.blur_slider.setValue(100)      # 默认100
        self.blur_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e74c3c;
                border: 1px solid #c0392b;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #c0392b;
            }
        """)
        self.blur_value_label = QLabel("100")
        self.blur_value_label.setStyleSheet("font-weight: bold; color: #e74c3c; min-width: 40px;")
        self.blur_value_label.setAlignment(Qt.AlignCenter)
        
        self.blur_slider.valueChanged.connect(self.update_blur_threshold)
        
        blur_control_layout.addWidget(self.blur_slider, 1)
        blur_control_layout.addWidget(self.blur_value_label)
        blur_layout.addLayout(blur_control_layout)
        
        # 阈值说明
        blur_desc = QLabel("经验值：方差 < 100 通常认为是模糊图片")
        blur_desc.setStyleSheet("font-size: 18px; color: #7f8c8d; margin-top: 5px;")
        blur_layout.addWidget(blur_desc)
        
        threshold_layout.addLayout(blur_layout)
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # 处理控制区域
        control_group = QGroupBox("处理控制")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        control_layout = QVBoxLayout()
        
        # 开始处理按钮
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_btn)
        
        # 停止处理按钮
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        control_layout.addWidget(self.stop_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 进度显示区域
        progress_group = QGroupBox("处理进度")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        progress_layout = QVBoxLayout()
        
        # 当前阶段显示
        self.stage_label = QLabel("等待开始...")
        self.stage_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        progress_layout.addWidget(self.stage_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 6px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
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
                border-radius: 5px;
                background-color: #ffffff;
                color: #000000;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 16px;
                padding: 10px;
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
    
    def update_blur_threshold(self, value):
        """更新模糊检测阈值"""
        self.blur_threshold = float(value)
        self.blur_value_label.setText(str(value))
    
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
        
        # 检查输出文件夹是否为输入文件夹的子文件夹
        if self.output_folder_path.startswith(self.input_folder_path):
            QMessageBox.warning(self, "警告", "输出文件夹不能是输入文件夹的子文件夹！")
            return
        
        # 创建输出文件夹
        try:
            os.makedirs(self.output_folder_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法创建输出文件夹: {str(e)}")
            return
        
        # 更新界面状态
        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # 启动处理线程
        self.processing_thread = ImageProcessingThread(
            self.input_folder_path,
            self.output_folder_path,
            self.blur_threshold,
            self
        )
        self.processing_thread.start()
        
        self.append_log("开始图像模糊检测处理...")
    
    def stop_processing(self):
        """停止处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.should_stop = True
            self.append_log("正在停止处理...")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_stage(self, stage):
        """更新当前阶段"""
        self.stage_label.setText(stage)
    
    def append_log(self, message):
        """添加日志信息"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
    
    def on_processing_finished(self, result):
        """处理完成回调"""
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if result['success']:
            self.append_log("处理完成！")
            self.append_log(f"总图片数: {result.get('total_images', 0)} 张")
            self.append_log(f"清晰图片: {result.get('clear_images', 0)} 张")
            self.append_log(f"模糊图片: {result.get('blurry_images', 0)} 张")
            self.append_log(f"处理时间: {result.get('processing_time', '未知')}")
            
            QMessageBox.information(self, "完成", 
                f"图像模糊检测完成！\n\n"
                f"总图片数: {result.get('total_images', 0)} 张\n"
                f"清晰图片: {result.get('clear_images', 0)} 张\n"
                f"模糊图片: {result.get('blurry_images', 0)} 张\n"
                f"处理时间: {result.get('processing_time', '未知')}")
        else:
            error_msg = result.get('error', '未知错误')
            self.append_log(f"处理失败: {error_msg}")
            QMessageBox.critical(self, "错误", f"处理失败: {error_msg}")
        
        self.update_stage("处理完成")
        self.progress_bar.setValue(100 if result['success'] else 0)


class ImageProcessingThread(QThread):
    """图像处理线程"""
    
    def __init__(self, input_folder, output_folder, blur_threshold, parent_dialog):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.blur_threshold = blur_threshold
        self.parent_dialog = parent_dialog
        self.should_stop = False
    
    def run(self):
        """线程主函数"""
        try:
            start_time = time.time()
            
            # 阶段1: 扫描图片文件
            self.parent_dialog.stage_updated.emit("阶段 1/3: 扫描图片文件...")
            self.parent_dialog.progress_updated.emit(5)
            
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            image_paths = []
            
            for root, _, files in os.walk(self.input_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(root, file))
            
            if not image_paths:
                self.parent_dialog.processing_finished.emit({
                    'success': False,
                    'error': '在输入文件夹中未找到任何图片文件'
                })
                return
            
            self.parent_dialog.log_updated.emit(f"找到 {len(image_paths)} 张图片")
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 阶段2: 模糊检测
            self.parent_dialog.stage_updated.emit("阶段 2/3: 模糊检测分析...")
            self.parent_dialog.progress_updated.emit(10)
            
            clear_images = []
            blurry_images = []
            
            for i, image_path in enumerate(image_paths):
                if self.should_stop:
                    self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                    return
                
                try:
                    # 计算Laplace方差
                    blur_score = self.calculate_laplace_variance(image_path)
                    
                    if blur_score >= self.blur_threshold:
                        clear_images.append((image_path, blur_score))
                    else:
                        blurry_images.append((image_path, blur_score))
                    
                    # 更新进度
                    progress = 10 + int((i + 1) / len(image_paths) * 70)
                    self.parent_dialog.progress_updated.emit(progress)
                    
                    if (i + 1) % 50 == 0:
                        self.parent_dialog.log_updated.emit(f"已检测 {i + 1}/{len(image_paths)} 张图片")
                
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"处理图片失败 {image_path}: {str(e)}")
                    continue
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 阶段3: 保存清晰图片
            self.parent_dialog.stage_updated.emit("阶段 3/3: 保存清晰图片...")
            self.parent_dialog.progress_updated.emit(80)
            
            # 创建输出子文件夹
            clear_folder = os.path.join(self.output_folder, "clear_images")
            blurry_folder = os.path.join(self.output_folder, "blurry_images")
            
            os.makedirs(clear_folder, exist_ok=True)
            os.makedirs(blurry_folder, exist_ok=True)
            
            # 复制清晰图片
            for i, (image_path, score) in enumerate(clear_images):
                if self.should_stop:
                    self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                    return
                
                try:
                    filename = os.path.basename(image_path)
                    # 保持原始文件名，处理重名冲突
                    output_path = os.path.join(clear_folder, filename)
                    counter = 1
                    while os.path.exists(output_path):
                        name_part, ext_part = os.path.splitext(filename)
                        output_path = os.path.join(clear_folder, f"{name_part}({counter}){ext_part}")
                        counter += 1
                    shutil.copy2(image_path, output_path)
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"复制清晰图片失败 {image_path}: {str(e)}")
                
                # 更新进度
                if clear_images:
                    progress = 80 + int((i + 1) / len(clear_images) * 10)
                    self.parent_dialog.progress_updated.emit(progress)
            
            # 复制模糊图片（可选）
            for i, (image_path, score) in enumerate(blurry_images[:50]):  # 最多保存50张模糊图片作为参考
                if self.should_stop:
                    break
                
                try:
                    filename = os.path.basename(image_path)
                    # 保持原始文件名，处理重名冲突
                    output_path = os.path.join(blurry_folder, filename)
                    counter = 1
                    while os.path.exists(output_path):
                        name_part, ext_part = os.path.splitext(filename)
                        output_path = os.path.join(blurry_folder, f"{name_part}({counter}){ext_part}")
                        counter += 1
                    shutil.copy2(image_path, output_path)
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"复制模糊图片失败 {image_path}: {str(e)}")
            
            # 生成报告
            self.generate_report(len(image_paths), len(clear_images), len(blurry_images), 
                               clear_folder, blurry_folder)
            
            end_time = time.time()
            processing_time = str(timedelta(seconds=int(end_time - start_time)))
            
            # 处理完成
            self.parent_dialog.processing_finished.emit({
                'success': True,
                'total_images': len(image_paths),
                'clear_images': len(clear_images),
                'blurry_images': len(blurry_images),
                'processing_time': processing_time
            })
            
        except Exception as e:
            self.parent_dialog.processing_finished.emit({
                'success': False,
                'error': str(e)
            })
    
    def calculate_laplace_variance(self, image_path):
        """计算图像的Laplace方差（模糊检测）"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 应用Laplace算子
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # 计算方差
            variance = laplacian.var()
            
            return variance
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"计算Laplace方差失败 {image_path}: {str(e)}")
            return 0.0
    
    def generate_report(self, total_images, clear_count, blurry_count, clear_folder, blurry_folder):
        """生成处理报告"""
        try:
            report_path = os.path.join(self.output_folder, "blur_detection_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("图像模糊检测报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入文件夹: {self.input_folder}\n")
                f.write(f"输出文件夹: {self.output_folder}\n")
                f.write(f"模糊检测阈值: {self.blur_threshold}\n\n")
                
                f.write("处理结果:\n")
                f.write("-" * 30 + "\n")
                f.write(f"总图片数量: {total_images} 张\n")
                f.write(f"清晰图片数量: {clear_count} 张 ({clear_count/total_images*100:.1f}%)\n")
                f.write(f"模糊图片数量: {blurry_count} 张 ({blurry_count/total_images*100:.1f}%)\n\n")
                
                f.write("输出说明:\n")
                f.write("-" * 30 + "\n")
                f.write(f"清晰图片保存在: {clear_folder}\n")
                f.write(f"模糊图片样本保存在: {blurry_folder}\n")
                f.write("文件名格式: 原文件名_clear/blurry_分数.扩展名\n\n")
                
                f.write("算法说明:\n")
                f.write("-" * 30 + "\n")
                f.write("使用Laplace方差法进行模糊检测\n")
                f.write("原理: 拉普拉斯算子提取边缘，计算边缘强度方差\n")
                f.write("模糊图片的高频信息少，方差值较低\n")
                f.write(f"阈值设置: 方差 < {self.blur_threshold} 认为是模糊图片\n")
            
            self.parent_dialog.log_updated.emit(f"报告已生成: {report_path}")
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"生成报告失败: {str(e)}")


def show_image_deblur_detection_dialog(parent=None):
    """显示图像去模糊检测对话框"""
    dialog = ImageDeblurDetectionDialog(parent)
    return dialog.exec_()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    dialog = ImageDeblurDetectionDialog()
    dialog.show()
    sys.exit(app.exec_())