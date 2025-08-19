#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型反标注对话框
基于YOLO模型自动生成图片标注文件
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QLineEdit, 
                            QProgressBar, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ModelInferenceThread(QThread):
    """模型推理线程类 - 在后台执行YOLO模型推理"""
    progress_updated = pyqtSignal(int)  # 进度更新信号
    log_updated = pyqtSignal(str)      # 日志更新信号
    finished = pyqtSignal(str)         # 完成信号
    
    def __init__(self, model_path, image_folder, save_label_folder):
        super().__init__()
        self.model_path = model_path          # YOLO模型文件路径
        self.image_folder = image_folder      # 图片文件夹路径
        self.save_label_folder = save_label_folder  # 标注文件保存路径
        
    def run(self):
        """线程主执行函数"""
        try:
            # 检查YOLO是否可用
            if not YOLO_AVAILABLE:
                self.finished.emit("错误：未安装ultralytics库，请先安装：pip install ultralytics")
                return
                
            self.log_updated.emit("正在加载YOLO模型...")
            # 加载YOLO模型
            model = YOLO(self.model_path)
            self.log_updated.emit(f"模型加载成功：{self.model_path}")
            
            # 获取所有图片文件（包括子文件夹中的图片）
            self.log_updated.emit("正在扫描图片文件...")
            image_files = []
            # 支持的图片格式
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            
            for ext in image_extensions:
                # 使用glob递归搜索所有子文件夹
                found_files = list(Path(self.image_folder).glob(f"**/{ext}"))
                image_files.extend(found_files)
                # 同时搜索大写扩展名
                found_files_upper = list(Path(self.image_folder).glob(f"**/{ext.upper()}"))
                image_files.extend(found_files_upper)
            
            # 去除重复文件
            image_files = list(set(image_files))
            total_files = len(image_files)
            
            if total_files == 0:
                self.finished.emit("没有找到图片文件，请检查文件夹路径")
                return
                
            self.log_updated.emit(f"找到 {total_files} 个图片文件")
            
            # 创建labels子文件夹
            labels_folder = Path(self.save_label_folder) / "labels"
            os.makedirs(labels_folder, exist_ok=True)
            self.log_updated.emit(f"标注文件将保存到：{labels_folder}")
            
            # 处理每张图片
            processed_count = 0
            for i, img_path in enumerate(image_files):
                try:
                    self.log_updated.emit(f"正在处理：{img_path.name}")
                    
                    # 运行YOLO推理
                    results = model(str(img_path))
                    
                    # 将标签文件保存到labels子文件夹中，保持与图片相同的名称
                    label_path = labels_folder / f"{img_path.stem}.txt"
                    
                    # 写入检测结果到标签文件
                    detection_count = 0
                    with open(label_path, 'w', encoding='utf-8') as f:
                        for result in results:
                            if result.boxes is not None:
                                boxes = result.boxes
                                for box in boxes:
                                    # 获取类别、置信度和归一化坐标
                                    cls = int(box.cls.item())
                                    conf = box.conf.item()
                                    x, y, w, h = box.xywhn[0].tolist()  # 归一化坐标
                                    
                                    # 按YOLO格式写入: class x_center y_center width height
                                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                                    detection_count += 1
                    
                    processed_count += 1
                    self.log_updated.emit(f"  检测到 {detection_count} 个目标")
                    
                except Exception as e:
                    self.log_updated.emit(f"  处理失败：{str(e)}")
                    continue
                
                # 更新进度
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
            
            self.finished.emit(f"处理完成！成功处理 {processed_count}/{total_files} 个文件，标注文件已保存到 labels 文件夹")
            
        except Exception as e:
            self.finished.emit(f"处理过程中出错：{str(e)}")


class ModelInferenceDialog(QDialog):
    """模型反标注对话框类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型反标注工具")
        self.setMinimumSize(700, 500)
        self.setModal(True)  # 设置为模态对话框
        
        # 初始化界面
        self.init_ui()
        
        # 线程对象
        self.inference_thread = None
        
    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title_label = QLabel("YOLO模型自动标注工具")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 说明文字
        desc_label = QLabel("使用训练好的YOLO模型对图片进行自动标注，生成YOLO格式的标注文件")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        main_layout.addWidget(desc_label)
        
        # 模型路径选择区域
        model_group = self.create_path_selection_group(
            "YOLO模型文件", 
            "选择模型文件 (.pt)", 
            "模型文件 (*.pt *.pth);;所有文件 (*.*)"
        )
        self.model_path_edit = model_group['edit']
        main_layout.addLayout(model_group['layout'])
        
        # 图片文件夹选择区域
        image_group = self.create_folder_selection_group(
            "图片文件夹", 
            "选择图片文件夹"
        )
        self.image_folder_edit = image_group['edit']
        main_layout.addLayout(image_group['layout'])
        
        # 输出文件夹选择区域
        output_group = self.create_folder_selection_group(
            "输出文件夹", 
            "选择输出文件夹"
        )
        self.output_folder_edit = output_group['edit']
        main_layout.addLayout(output_group['layout'])
        
        # 进度条
        progress_layout = QVBoxLayout()
        progress_label = QLabel("处理进度:")
        progress_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)
        
        # 日志显示区域
        log_layout = QVBoxLayout()
        log_label = QLabel("处理日志:")
        log_label.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
            }
        """)
        log_layout.addWidget(self.log_text)
        main_layout.addLayout(log_layout)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setMinimumSize(100, 35)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_inference)
        button_layout.addWidget(self.start_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.setMinimumSize(100, 35)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        main_layout.addLayout(button_layout)
        
        # 检查YOLO是否可用
        if not YOLO_AVAILABLE:
            self.log_text.append("⚠️ 警告：未检测到ultralytics库，请先安装：pip install ultralytics")
            self.start_btn.setEnabled(False)
        else:
            self.log_text.append("✅ YOLO环境检查通过，可以开始使用")
    
    def create_path_selection_group(self, label_text, button_text, file_filter):
        """创建文件路径选择组件"""
        layout = QVBoxLayout()
        
        # 标签
        label = QLabel(label_text + ":")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        
        # 路径输入和按钮的水平布局
        h_layout = QHBoxLayout()
        
        # 路径输入框
        path_edit = QLineEdit()
        path_edit.setPlaceholderText(f"请选择{label_text}...")
        h_layout.addWidget(path_edit)
        
        # 选择按钮
        select_btn = QPushButton(button_text)
        select_btn.setMinimumWidth(120)
        select_btn.clicked.connect(lambda: self.select_file(path_edit, file_filter))
        h_layout.addWidget(select_btn)
        
        layout.addLayout(h_layout)
        
        return {'layout': layout, 'edit': path_edit}
    
    def create_folder_selection_group(self, label_text, button_text):
        """创建文件夹选择组件"""
        layout = QVBoxLayout()
        
        # 标签
        label = QLabel(label_text + ":")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        
        # 路径输入和按钮的水平布局
        h_layout = QHBoxLayout()
        
        # 路径输入框
        folder_edit = QLineEdit()
        folder_edit.setPlaceholderText(f"请选择{label_text}...")
        h_layout.addWidget(folder_edit)
        
        # 选择按钮
        select_btn = QPushButton(button_text)
        select_btn.setMinimumWidth(120)
        select_btn.clicked.connect(lambda: self.select_folder(folder_edit))
        h_layout.addWidget(select_btn)
        
        layout.addLayout(h_layout)
        
        return {'layout': layout, 'edit': folder_edit}
    
    def select_file(self, line_edit, file_filter):
        """选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)
    
    def select_folder(self, line_edit):
        """选择文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            line_edit.setText(folder_path)
    
    def start_inference(self):
        """开始模型推理"""
        # 获取用户输入的路径
        model_path = self.model_path_edit.text().strip()
        image_folder = self.image_folder_edit.text().strip()
        output_folder = self.output_folder_edit.text().strip()
        
        # 验证输入
        if not all([model_path, image_folder, output_folder]):
            QMessageBox.warning(self, "输入错误", "请填写所有必要的路径！")
            return
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "路径错误", "模型文件不存在！")
            return
            
        if not os.path.exists(image_folder):
            QMessageBox.warning(self, "路径错误", "图片文件夹不存在！")
            return
            
        if not os.path.exists(output_folder):
            QMessageBox.warning(self, "路径错误", "输出文件夹不存在！")
            return
        
        # 禁用开始按钮，防止重复点击
        self.start_btn.setEnabled(False)
        self.start_btn.setText("处理中...")
        
        # 重置进度条和日志
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_text.append("开始模型反标注处理...")
        
        # 创建并启动推理线程
        self.inference_thread = ModelInferenceThread(model_path, image_folder, output_folder)
        self.inference_thread.progress_updated.connect(self.update_progress)
        self.inference_thread.log_updated.connect(self.update_log)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_log(self, message):
        """更新日志显示"""
        self.log_text.append(message)
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def on_inference_finished(self, message):
        """推理完成回调"""
        self.update_log(message)
        
        # 恢复开始按钮
        self.start_btn.setEnabled(True)
        self.start_btn.setText("开始处理")
        
        # 显示完成消息
        if "处理完成" in message:
            QMessageBox.information(self, "处理完成", message)
        else:
            QMessageBox.warning(self, "处理失败", message)
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 如果线程正在运行，询问用户是否确认关闭
        if self.inference_thread and self.inference_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认关闭", 
                "模型推理正在进行中，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # 终止线程
                self.inference_thread.terminate()
                self.inference_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()