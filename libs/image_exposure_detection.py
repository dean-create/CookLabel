#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像曝光检测工具 - 基于灰度直方图的曝光异常检测
支持用户图形化选择输入文件夹、输出文件夹，并自定义曝光检测阈值
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


class ImageExposureDetectionDialog(QDialog):
    """图像曝光检测工具对话框"""
    
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
        super(ImageExposureDetectionDialog, self).__init__(parent)
        self.setWindowTitle("图像曝光检测工具")
        self.setWindowIcon(new_icon('app'))
        self.setMinimumSize(1000, 750)  # 设置最小尺寸
        self.resize(1200, 850)          # 设置默认尺寸
        
        # 初始化变量
        self.input_folder_path = ""
        self.output_folder_path = ""
        self.is_processing = False      # 是否正在处理
        self.processing_thread = None   # 处理线程
        
        # 曝光检测阈值参数
        self.underexposure_threshold = 80.0  # 欠曝阈值：>80%像素在0-50亮度范围
        self.overexposure_threshold = 80.0   # 过曝阈值：>80%像素在200-255亮度范围
        self.low_brightness_range = (0, 50)  # 低亮度范围
        self.high_brightness_range = (200, 255)  # 高亮度范围
        
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
        title_label = QLabel("图像曝光检测工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; margin: 15px; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # 功能说明
        description_label = QLabel("基于灰度直方图分析检测欠曝和过曝图片，自动筛选正常曝光图像")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("font-size: 16px; color: #7f8c8d; margin-bottom: 10px;")
        layout.addWidget(description_label)
        
        # 路径选择区域
        path_group = QGroupBox("路径设置")
        path_layout = QVBoxLayout()
        
        # 输入图片文件夹选择
        input_folder_label_title = QLabel("输入图片文件夹:")
        input_folder_label_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        path_layout.addWidget(input_folder_label_title)
        
        input_folder_layout = QHBoxLayout()
        self.input_path_label = QLabel("请选择包含图片的文件夹...")
        self.input_path_label.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        self.input_path_label.setWordWrap(True)  # 允许文本换行
        input_button = QPushButton("浏览")
        input_button.setFixedWidth(80)  # 固定按钮宽度
        input_button.clicked.connect(self.select_input_folder)
        input_folder_layout.addWidget(self.input_path_label, 1)
        input_folder_layout.addWidget(input_button)
        path_layout.addLayout(input_folder_layout)
        
        # 添加间距
        path_layout.addSpacing(10)
        
        # 输出文件夹选择
        output_folder_label_title = QLabel("输出文件夹:")
        output_folder_label_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        path_layout.addWidget(output_folder_label_title)
        
        output_folder_layout = QHBoxLayout()
        self.output_path_label = QLabel("请选择输出文件夹...")
        self.output_path_label.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        self.output_path_label.setWordWrap(True)  # 允许文本换行
        output_button = QPushButton("浏览")
        output_button.setFixedWidth(80)  # 固定按钮宽度
        output_button.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(self.output_path_label, 1)
        output_folder_layout.addWidget(output_button)
        path_layout.addLayout(output_folder_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 曝光检测参数设置区域
        params_group = QGroupBox("曝光检测参数")
        params_layout = QVBoxLayout()
        
        # 欠曝阈值设置
        under_layout = QHBoxLayout()
        under_label = QLabel("欠曝检测阈值 (%)：")
        under_label.setMinimumWidth(150)
        self.under_slider = QSlider(Qt.Horizontal)
        self.under_slider.setMinimum(50)
        self.under_slider.setMaximum(95)
        self.under_slider.setValue(int(self.underexposure_threshold))
        self.under_slider.valueChanged.connect(self.update_underexposure_threshold)
        self.under_value_label = QLabel(str(int(self.underexposure_threshold)))
        self.under_value_label.setMinimumWidth(30)
        under_layout.addWidget(under_label)
        under_layout.addWidget(self.under_slider, 1)
        under_layout.addWidget(self.under_value_label)
        params_layout.addLayout(under_layout)
        
        # 过曝阈值设置
        over_layout = QHBoxLayout()
        over_label = QLabel("过曝检测阈值 (%)：")
        over_label.setMinimumWidth(150)
        self.over_slider = QSlider(Qt.Horizontal)
        self.over_slider.setMinimum(50)
        self.over_slider.setMaximum(95)
        self.over_slider.setValue(int(self.overexposure_threshold))
        self.over_slider.valueChanged.connect(self.update_overexposure_threshold)
        self.over_value_label = QLabel(str(int(self.overexposure_threshold)))
        self.over_value_label.setMinimumWidth(30)
        over_layout.addWidget(over_label)
        over_layout.addWidget(self.over_slider, 1)
        over_layout.addWidget(self.over_value_label)
        params_layout.addLayout(over_layout)
        
        # 参数说明
        params_info = QLabel("说明：欠曝 - >阈值%像素在0-50亮度范围；过曝 - >阈值%像素在200-255亮度范围")
        params_info.setStyleSheet("font-size: 16px; color: #7f8c8d; margin-top: 5px;")
        params_layout.addWidget(params_info)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 处理控制区域
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始检测")
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; border-radius: 6px; } QPushButton:hover { background-color: #2980b9; }")
        self.start_button.clicked.connect(self.start_processing)
        
        self.stop_button = QPushButton("停止处理")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; border-radius: 6px; } QPushButton:hover { background-color: #c0392b; }")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        layout.addLayout(control_layout)
        
        # 进度显示区域
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        
        # 当前阶段显示
        self.stage_label = QLabel("等待开始处理...")
        self.stage_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-bottom: 5px;")
        progress_layout.addWidget(self.stage_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { border: 2px solid #bdc3c7; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }")
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 日志显示区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 16px;
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
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    def select_input_folder(self):
        """选择输入文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        if folder:
            self.input_folder_path = folder
            self.input_path_label.setText(folder)
            self.append_log(f"已选择输入文件夹: {folder}")
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_path = folder
            self.output_path_label.setText(folder)
            self.append_log(f"已选择输出文件夹: {folder}")
    
    def update_underexposure_threshold(self, value):
        """更新欠曝检测阈值"""
        self.underexposure_threshold = float(value)
        self.under_value_label.setText(str(value))
    
    def update_overexposure_threshold(self, value):
        """更新过曝检测阈值"""
        self.overexposure_threshold = float(value)
        self.over_value_label.setText(str(value))
    
    def start_processing(self):
        """开始处理"""
        # 检查输入参数
        if not self.input_folder_path:
            QMessageBox.warning(self, "警告", "请先选择输入文件夹！")
            return
        
        if not self.output_folder_path:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return
        
        if not os.path.exists(self.input_folder_path):
            QMessageBox.critical(self, "错误", "输入文件夹不存在！")
            return
        
        # 创建输出文件夹
        try:
            os.makedirs(self.output_folder_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建输出文件夹失败：{str(e)}")
            return
        
        # 更新界面状态
        self.is_processing = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # 启动处理线程
        self.processing_thread = ImageProcessingThread(
            self.input_folder_path,
            self.output_folder_path,
            self.underexposure_threshold,
            self.overexposure_threshold,
            self.low_brightness_range,
            self.high_brightness_range,
            self
        )
        self.processing_thread.start()
        
        self.append_log("开始图像曝光检测处理...")
    
    def stop_processing(self):
        """停止处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.should_stop = True
            self.append_log("正在停止处理，请稍候...")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_stage(self, stage_text):
        """更新当前处理阶段"""
        self.stage_label.setText(stage_text)
    
    def append_log(self, message):
        """添加日志信息"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_processing_finished(self, result):
        """处理完成回调"""
        self.is_processing = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if result.get('success', False):
            stats = result.get('stats', {})
            normal_count = stats.get('normal_count', 0)
            underexposed_count = stats.get('underexposed_count', 0)
            overexposed_count = stats.get('overexposed_count', 0)
            total_count = normal_count + underexposed_count + overexposed_count
            
            self.append_log("图像曝光检测处理完成！")
            self.append_log(f"总处理图片: {total_count} 张")
            self.append_log(f"正常曝光图片: {normal_count} 张")
            self.append_log(f"欠曝图片: {underexposed_count} 张")
            self.append_log(f"过曝图片: {overexposed_count} 张")
            
            if total_count > 0:
                normal_rate = (normal_count / total_count) * 100
                self.append_log(f"正常曝光率: {normal_rate:.2f}%")
            
            # 显示完成对话框
            QMessageBox.information(
                self, "处理完成",
                f"图像曝光检测处理完成！\n\n"
                f"总处理图片: {total_count} 张\n"
                f"正常曝光图片: {normal_count} 张\n"
                f"欠曝图片: {underexposed_count} 张\n"
                f"过曝图片: {overexposed_count} 张\n\n"
                f"结果已保存到: {self.output_folder_path}"
            )
        else:
            error_msg = result.get('error', '未知错误')
            self.append_log(f"处理失败: {error_msg}")
            QMessageBox.warning(self, "错误", f"图像曝光检测处理失败:\n{error_msg}")
        
        self.update_stage("处理完成")
        self.progress_bar.setValue(100 if result.get('success', False) else 0)


class ImageProcessingThread(QThread):
    """图像处理线程"""
    
    def __init__(self, input_folder, output_folder, under_threshold, over_threshold, 
                 low_range, high_range, parent_dialog):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.under_threshold = under_threshold
        self.over_threshold = over_threshold
        self.low_range = low_range
        self.high_range = high_range
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
            
            # 阶段2: 曝光检测分析
            self.parent_dialog.stage_updated.emit("阶段 2/3: 曝光检测分析...")
            self.parent_dialog.progress_updated.emit(10)
            
            normal_images = []      # 正常曝光图片
            underexposed_images = []  # 欠曝图片
            overexposed_images = []   # 过曝图片
            
            for i, image_path in enumerate(image_paths):
                if self.should_stop:
                    self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                    return
                
                try:
                    # 分析图像曝光情况
                    exposure_type, exposure_score = self.analyze_exposure(image_path)
                    
                    if exposure_type == 'normal':
                        normal_images.append((image_path, exposure_score))
                    elif exposure_type == 'underexposed':
                        underexposed_images.append((image_path, exposure_score))
                    elif exposure_type == 'overexposed':
                        overexposed_images.append((image_path, exposure_score))
                    
                    # 更新进度
                    progress = 10 + int((i + 1) / len(image_paths) * 70)
                    self.parent_dialog.progress_updated.emit(progress)
                    
                    # 每处理50张图片输出一次进度
                    if (i + 1) % 50 == 0:
                        self.parent_dialog.log_updated.emit(f"已分析 {i + 1}/{len(image_paths)} 张图片")
                
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"分析图片失败 {image_path}: {str(e)}")
                    continue
            
            if self.should_stop:
                self.parent_dialog.processing_finished.emit({'success': False, 'error': '用户中断处理'})
                return
            
            # 阶段3: 保存分类结果
            self.parent_dialog.stage_updated.emit("阶段 3/3: 保存分类结果...")
            self.parent_dialog.progress_updated.emit(80)
            
            # 创建输出子文件夹
            normal_folder = os.path.join(self.output_folder, "normal_exposure")
            under_folder = os.path.join(self.output_folder, "underexposed")
            over_folder = os.path.join(self.output_folder, "overexposed")
            
            os.makedirs(normal_folder, exist_ok=True)
            os.makedirs(under_folder, exist_ok=True)
            os.makedirs(over_folder, exist_ok=True)
            
            # 保存正常曝光图片
            self.save_images(normal_images, normal_folder, "normal")
            
            # 保存欠曝图片
            self.save_images(underexposed_images, under_folder, "under")
            
            # 保存过曝图片
            self.save_images(overexposed_images, over_folder, "over")
            
            # 生成处理报告
            self.generate_report(normal_images, underexposed_images, overexposed_images)
            
            # 计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            time_str = str(timedelta(seconds=int(processing_time)))
            
            self.parent_dialog.log_updated.emit("图像曝光检测处理完成！")
            self.parent_dialog.log_updated.emit(f"处理时间: {time_str}")
            self.parent_dialog.log_updated.emit(f"正常曝光图片: {len(normal_images)} 张")
            self.parent_dialog.log_updated.emit(f"欠曝图片: {len(underexposed_images)} 张")
            self.parent_dialog.log_updated.emit(f"过曝图片: {len(overexposed_images)} 张")
            self.parent_dialog.log_updated.emit(f"正常曝光输出目录: {normal_folder}")
            self.parent_dialog.log_updated.emit(f"欠曝图片输出目录: {under_folder}")
            self.parent_dialog.log_updated.emit(f"过曝图片输出目录: {over_folder}")
            
            # 返回处理结果
            self.parent_dialog.processing_finished.emit({
                'success': True,
                'stats': {
                    'normal_count': len(normal_images),
                    'underexposed_count': len(underexposed_images),
                    'overexposed_count': len(overexposed_images),
                    'processing_time': time_str
                }
            })
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"处理过程中发生错误: {str(e)}")
            self.parent_dialog.processing_finished.emit({
                'success': False,
                'error': str(e)
            })
    
    def analyze_exposure(self, image_path):
        """
        分析图像曝光情况
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            tuple: (曝光类型, 曝光分数)
                曝光类型: 'normal', 'underexposed', 'overexposed'
                曝光分数: 低亮度像素百分比或高亮度像素百分比
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return 'normal', 0.0
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 计算灰度直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # 计算总像素数
            total_pixels = gray.shape[0] * gray.shape[1]
            
            # 计算低亮度范围像素数量（0-50）
            low_pixels = np.sum(hist[self.low_range[0]:self.low_range[1]+1])
            low_percentage = (low_pixels / total_pixels) * 100
            
            # 计算高亮度范围像素数量（200-255）
            high_pixels = np.sum(hist[self.high_range[0]:self.high_range[1]+1])
            high_percentage = (high_pixels / total_pixels) * 100
            
            # 判断曝光类型
            if low_percentage > self.under_threshold:
                return 'underexposed', low_percentage
            elif high_percentage > self.over_threshold:
                return 'overexposed', high_percentage
            else:
                return 'normal', max(low_percentage, high_percentage)
                
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"分析曝光失败 {image_path}: {str(e)}")
            return 'normal', 0.0
    
    def save_images(self, image_list, output_folder, prefix):
        """
        保存图片到指定文件夹
        
        Args:
            image_list: 图片列表，每个元素为(图片路径, 曝光分数)
            output_folder: 输出文件夹
            prefix: 文件名前缀
        """
        for i, (image_path, score) in enumerate(image_list):
            if self.should_stop:
                return
            
            try:
                # 保持原始文件名，处理重名冲突
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_folder, filename)
                counter = 1
                while os.path.exists(output_path):
                    name_part, ext_part = os.path.splitext(filename)
                    output_path = os.path.join(output_folder, f"{name_part}({counter}){ext_part}")
                    counter += 1
                
                # 复制文件
                shutil.copy2(image_path, output_path)
                
                # 每保存10张图片输出一次进度
                if (i + 1) % 10 == 0:
                    self.parent_dialog.log_updated.emit(f"已保存 {i + 1}/{len(image_list)} 张{prefix}图片")
                    
            except Exception as e:
                self.parent_dialog.log_updated.emit(f"保存{prefix}图片失败: {e}")
    
    def generate_report(self, normal_images, underexposed_images, overexposed_images):
        """
        生成处理报告
        
        Args:
            normal_images: 正常曝光图片列表
            underexposed_images: 欠曝图片列表
            overexposed_images: 过曝图片列表
        """
        try:
            report_path = os.path.join(self.output_folder, "exposure_detection_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("图像曝光检测处理报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 基本统计信息
                total_count = len(normal_images) + len(underexposed_images) + len(overexposed_images)
                f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总处理图片: {total_count} 张\n")
                f.write(f"正常曝光图片: {len(normal_images)} 张\n")
                f.write(f"欠曝图片: {len(underexposed_images)} 张\n")
                f.write(f"过曝图片: {len(overexposed_images)} 张\n\n")
                
                # 百分比统计
                if total_count > 0:
                    normal_rate = (len(normal_images) / total_count) * 100
                    under_rate = (len(underexposed_images) / total_count) * 100
                    over_rate = (len(overexposed_images) / total_count) * 100
                    
                    f.write(f"正常曝光率: {normal_rate:.2f}%\n")
                    f.write(f"欠曝率: {under_rate:.2f}%\n")
                    f.write(f"过曝率: {over_rate:.2f}%\n\n")
                
                # 检测参数
                f.write("检测参数:\n")
                f.write(f"欠曝阈值: {self.under_threshold}% (像素在0-50亮度范围)\n")
                f.write(f"过曝阈值: {self.over_threshold}% (像素在200-255亮度范围)\n\n")
                
                # 输出说明
                f.write("输出说明:\n")
                f.write("- normal_exposure文件夹包含正常曝光的图片\n")
                f.write("- underexposed文件夹包含欠曝的图片\n")
                f.write("- overexposed文件夹包含过曝的图片\n")
                f.write("- 文件名格式：原名_类型_score_分数值.扩展名\n")
                f.write("- 分数值表示低亮度或高亮度像素的百分比\n\n")
                
                f.write("建议：\n")
                f.write("- 根据实际需求调整检测阈值\n")
                f.write("- 人工抽查结果，确认检测效果\n")
                f.write("- 可以进一步对异常曝光图片进行后处理\n")
            
            self.parent_dialog.log_updated.emit(f"处理报告已保存: {report_path}")
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"生成报告失败: {str(e)}")


def show_image_exposure_detection_dialog(parent=None):
    """显示图像曝光检测对话框"""
    dialog = ImageExposureDetectionDialog(parent)
    return dialog.exec_()