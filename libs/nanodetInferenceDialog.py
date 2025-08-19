#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NanoDet模型反标注对话框
为CookLabel提供NanoDet模型自动标注功能的图形界面
支持模型选择、参数配置和批量处理
"""

import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar,
    QTextEdit, QGroupBox, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QMessageBox, QFileDialog,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap

# 导入NanoDet推理模块
try:
    from .nanodet_inference import (
        NanoDetInference, NanoDetConfig, create_nanodet_inference
    )
    NANODET_AVAILABLE = True
except ImportError as e:
    NANODET_AVAILABLE = False
    print(f"警告: NanoDet推理模块导入失败 - {str(e)}")


class NanoDetInferenceThread(QThread):
    """
    NanoDet模型推理线程
    在后台执行模型推理任务，避免阻塞主界面
    """
    
    # 定义信号
    progress_updated = pyqtSignal(int)  # 进度更新信号
    log_updated = pyqtSignal(str)  # 日志更新信号
    finished = pyqtSignal(dict)  # 完成信号，传递统计信息
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, model_path, input_dir, output_dir, config):
        """
        初始化推理线程
        
        参数:
            model_path: 模型文件路径
            input_dir: 输入图像目录
            output_dir: 输出标注目录
            config: NanoDet配置对象
        """
        super().__init__()
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.is_cancelled = False
        
    def cancel(self):
        """取消推理任务"""
        self.is_cancelled = True
        
    def run(self):
        """
        执行推理任务的主方法
        """
        try:
            # 检查NanoDet模块可用性
            if not NANODET_AVAILABLE:
                self.error_occurred.emit("NanoDet模块不可用，请检查依赖安装")
                return
            
            # 发送开始日志
            self.log_updated.emit("开始初始化NanoDet模型...")
            
            # 创建推理器
            device = 'cuda' if self.config.use_gpu else 'cpu'
            inference = NanoDetInference(
                model_path=self.model_path,
                config=self.config,
                device=device
            )
            
            self.log_updated.emit(f"模型加载成功，使用设备: {device}")
            self.log_updated.emit(f"输入目录: {self.input_dir}")
            self.log_updated.emit(f"输出目录: {self.output_dir}")
            self.log_updated.emit(f"输出格式: {self.config.output_format.upper()}")
            self.log_updated.emit(f"置信度阈值: {self.config.confidence_threshold}")
            self.log_updated.emit("开始处理图像...")
            
            # 定义进度回调函数
            def progress_callback(progress):
                if not self.is_cancelled:
                    self.progress_updated.emit(progress)
                    self.log_updated.emit(f"处理进度: {progress}%")
            
            # 执行批量推理
            statistics = inference.process_images(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                progress_callback=progress_callback
            )
            
            if self.is_cancelled:
                self.log_updated.emit("任务已取消")
                return
            
            # 发送完成信号
            self.log_updated.emit("处理完成！")
            self.log_updated.emit(f"总计: {statistics['total']} 张图像")
            self.log_updated.emit(f"成功: {statistics['processed']} 张")
            self.log_updated.emit(f"失败: {statistics['failed']} 张")
            
            self.finished.emit(statistics)
            
        except Exception as e:
            error_msg = f"推理过程中发生错误: {str(e)}"
            self.log_updated.emit(error_msg)
            self.error_occurred.emit(error_msg)


class NanoDetInferenceDialog(QDialog):
    """
    NanoDet模型反标注对话框主类
    提供完整的图形界面用于配置和执行NanoDet模型推理
    """
    
    def __init__(self, parent=None):
        """
        初始化对话框
        
        参数:
            parent: 父窗口对象
        """
        super().__init__(parent)
        self.parent = parent
        self.inference_thread = None
        
        # 检查NanoDet模块是否可用
        if not NANODET_AVAILABLE:
            QMessageBox.critical(
                self, 
                "模块导入错误", 
                "NanoDet推理模块导入失败！\n\n可能的原因：\n1. 缺少NanoDet相关依赖\n2. 模型文件路径不正确\n3. Python环境配置问题\n\n请检查环境配置后重试。"
            )
            self.reject()  # 关闭对话框
            return
            
        self.config = NanoDetConfig()
        
        # 设置对话框属性
        self.setWindowTitle("NanoDet模型反标注")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(900, 850)  # 增大窗口尺寸，提高可读性
        
        # 初始化界面
        self.init_ui()
        
        # 连接信号槽
        self.connect_signals()
        
    def init_ui(self):
        """
        初始化用户界面
        创建所有界面组件并设置布局
        """
        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 创建标题
        title_label = QLabel("NanoDet模型自动标注")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # 创建配置区域
        config_group = self.create_config_group()
        main_layout.addWidget(config_group)
        
        # 创建高级参数区域
        advanced_group = self.create_advanced_group()
        main_layout.addWidget(advanced_group)
        
        # 创建进度区域
        progress_group = self.create_progress_group()
        main_layout.addWidget(progress_group)
        
        # 创建日志区域
        log_group = self.create_log_group()
        main_layout.addWidget(log_group)
        
        # 创建按钮区域
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)
        
        # 设置主布局
        self.setLayout(main_layout)
        
        # 应用样式
        self.apply_styles()
        
    def create_config_group(self):
        """
        创建基本配置组
        包含模型文件、输入目录、输出目录选择
        
        返回:
            config_group: 配置组控件
        """
        config_group = QGroupBox("基本配置")
        config_layout = QGridLayout()
        
        # 模型文件选择
        config_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择NanoDet模型文件 (.pth, .pt)")
        config_layout.addWidget(self.model_path_edit, 0, 1)
        
        self.model_browse_btn = QPushButton("浏览")
        self.model_browse_btn.clicked.connect(self.browse_model_file)
        config_layout.addWidget(self.model_browse_btn, 0, 2)
        
        # 输入目录选择
        config_layout.addWidget(QLabel("输入目录:"), 1, 0)
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("选择包含图像的输入目录")
        config_layout.addWidget(self.input_dir_edit, 1, 1)
        
        self.input_browse_btn = QPushButton("浏览")
        self.input_browse_btn.clicked.connect(self.browse_input_dir)
        config_layout.addWidget(self.input_browse_btn, 1, 2)
        
        # 输出目录选择
        config_layout.addWidget(QLabel("输出目录:"), 2, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择标注文件输出目录")
        config_layout.addWidget(self.output_dir_edit, 2, 1)
        
        self.output_browse_btn = QPushButton("浏览")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        config_layout.addWidget(self.output_browse_btn, 2, 2)
        
        config_group.setLayout(config_layout)
        return config_group
        
    def create_advanced_group(self):
        """
        创建高级参数配置组
        包含置信度阈值、输出格式、设备选择等参数
        
        返回:
            advanced_group: 高级参数组控件
        """
        advanced_group = QGroupBox("高级参数")
        advanced_layout = QGridLayout()
        
        # 置信度阈值
        advanced_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.01, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.35)
        self.confidence_spinbox.setDecimals(2)
        advanced_layout.addWidget(self.confidence_spinbox, 0, 1)
        
        # NMS阈值 - 添加详细说明
        nms_label = QLabel("NMS阈值:")
        nms_label.setToolTip("非极大值抑制阈值：用于去除重叠的检测框。\n值越小，去除重叠框越严格；值越大，保留更多检测框。\n推荐范围：0.3-0.7")
        advanced_layout.addWidget(nms_label, 0, 2)
        self.nms_spinbox = QDoubleSpinBox()
        self.nms_spinbox.setRange(0.01, 1.0)
        self.nms_spinbox.setSingleStep(0.05)
        self.nms_spinbox.setValue(0.6)
        self.nms_spinbox.setDecimals(2)
        self.nms_spinbox.setToolTip("非极大值抑制阈值：用于去除重叠的检测框。\n值越小，去除重叠框越严格；值越大，保留更多检测框。\n推荐范围：0.3-0.7")
        advanced_layout.addWidget(self.nms_spinbox, 0, 3)
        
        # 最大检测数量
        advanced_layout.addWidget(QLabel("最大检测数:"), 1, 0)
        self.max_det_spinbox = QSpinBox()
        self.max_det_spinbox.setRange(1, 1000)
        self.max_det_spinbox.setValue(100)
        advanced_layout.addWidget(self.max_det_spinbox, 1, 1)
        
        # 输入尺寸
        advanced_layout.addWidget(QLabel("输入尺寸:"), 1, 2)
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["416x416", "320x320", "512x512", "640x640"])
        self.input_size_combo.setCurrentText("416x416")
        advanced_layout.addWidget(self.input_size_combo, 1, 3)
        
        advanced_group.setLayout(advanced_layout)
        return advanced_group
        
    def create_progress_group(self):
        """
        创建进度显示组
        包含进度条和状态标签
        
        返回:
            progress_group: 进度组控件
        """
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        return progress_group
        
    def create_log_group(self):
        """
        创建日志显示组
        包含日志文本区域和清空按钮
        
        返回:
            log_group: 日志组控件
        """
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        # 日志文本区域
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        # 清空日志按钮
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_btn)
        
        log_group.setLayout(log_layout)
        return log_group
        
    def create_button_layout(self):
        """
        创建按钮布局
        包含开始处理、取消处理、关闭对话框按钮
        
        返回:
            button_layout: 按钮布局
        """
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 开始处理按钮
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setMinimumWidth(100)
        button_layout.addWidget(self.start_btn)
        
        # 取消处理按钮
        self.cancel_btn = QPushButton("取消处理")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumWidth(100)
        button_layout.addWidget(self.cancel_btn)
        
        # 关闭按钮
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setMinimumWidth(100)
        button_layout.addWidget(self.close_btn)
        
        return button_layout
        
    def apply_styles(self):
        """
        应用界面样式
        设置整体的视觉风格
        """
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #007bff;
                outline: none;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                background-color: #e9ecef;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 3px;
            }
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-family: 'Consolas', monospace;
            }
        """)
        
    def connect_signals(self):
        """
        连接信号和槽
        设置界面组件之间的交互逻辑
        """
        # 连接信号
        self.confidence_spinbox.valueChanged.connect(self.update_config)
        self.nms_spinbox.valueChanged.connect(self.update_config)
        self.max_det_spinbox.valueChanged.connect(self.update_config)
        self.input_size_combo.currentTextChanged.connect(self.update_config)
        
    def browse_model_file(self):
        """
        浏览选择模型文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择NanoDet模型文件", "",
            "模型文件 (*.pth *.pt);;所有文件 (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def browse_input_dir(self):
        """
        浏览选择输入目录
        """
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输入图像目录"
        )
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            
    def browse_output_dir(self):
        """
        浏览选择输出目录
        """
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出标注目录"
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            
    def update_config(self):
        """
        更新配置对象
        根据界面参数更新NanoDet配置
        """
        # 更新置信度阈值
        self.config.confidence_threshold = self.confidence_spinbox.value()
        
        # 更新NMS阈值
        self.config.nms_threshold = self.nms_spinbox.value()
        
        # 固定输出格式为YOLO
        self.config.output_format = 'yolo'
            
        # 固定推理设备为CPU
        self.config.use_gpu = False
        
        # 更新最大检测数量
        self.config.max_detections = self.max_det_spinbox.value()
        
        # 更新输入尺寸
        size_text = self.input_size_combo.currentText()
        if "x" in size_text:
            width, height = map(int, size_text.split("x"))
            self.config.input_size = (width, height)
            
    def validate_inputs(self):
        """
        验证输入参数
        
        返回:
            valid: 是否有效
            error_msg: 错误信息
        """
        # 检查模型文件
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            return False, "请选择模型文件"
        if not os.path.exists(model_path):
            return False, "模型文件不存在"
            
        # 检查输入目录
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            return False, "请选择输入目录"
        if not os.path.exists(input_dir):
            return False, "输入目录不存在"
            
        # 检查输出目录
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            return False, "请选择输出目录"
                
        return True, ""
        
    def start_processing(self):
        """
        开始处理任务
        """
        # 验证输入
        valid, error_msg = self.validate_inputs()
        if not valid:
            QMessageBox.warning(self, "输入错误", error_msg)
            return
            
        # 更新配置
        self.update_config()
        
        # 创建输出目录
        output_dir = self.output_dir_edit.text().strip()
        os.makedirs(output_dir, exist_ok=True)
        
        # 更新界面状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在处理...")
        
        # 清空日志
        self.log_text.clear()
        
        # 创建并启动推理线程
        self.inference_thread = NanoDetInferenceThread(
            model_path=self.model_path_edit.text().strip(),
            input_dir=self.input_dir_edit.text().strip(),
            output_dir=output_dir,
            config=self.config
        )
        
        # 连接线程信号
        self.inference_thread.progress_updated.connect(self.update_progress)
        self.inference_thread.log_updated.connect(self.update_log)
        self.inference_thread.finished.connect(self.processing_finished)
        self.inference_thread.error_occurred.connect(self.processing_error)
        
        # 启动线程
        self.inference_thread.start()
        
    def cancel_processing(self):
        """
        取消处理任务
        """
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.cancel()
            self.inference_thread.wait(3000)  # 等待3秒
            
            if self.inference_thread.isRunning():
                self.inference_thread.terminate()
                self.inference_thread.wait()
                
        self.reset_ui_state()
        self.status_label.setText("已取消")
        self.update_log("处理已取消")
        
    def processing_finished(self, statistics):
        """
        处理完成回调
        
        参数:
            statistics: 处理统计信息
        """
        self.reset_ui_state()
        self.progress_bar.setValue(100)
        self.status_label.setText("处理完成")
        
        # 显示完成对话框
        QMessageBox.information(
            self, "处理完成",
            f"处理完成！\n\n"
            f"总计: {statistics['total']} 张图像\n"
            f"成功: {statistics['processed']} 张\n"
            f"失败: {statistics['failed']} 张"
        )
        
    def processing_error(self, error_msg):
        """
        处理错误回调
        
        参数:
            error_msg: 错误信息
        """
        self.reset_ui_state()
        self.status_label.setText("处理失败")
        
        # 显示错误对话框
        QMessageBox.critical(self, "处理错误", error_msg)
        
    def reset_ui_state(self):
        """
        重置界面状态
        """
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
    def update_progress(self, progress):
        """
        更新进度条
        
        参数:
            progress: 进度值 (0-100)
        """
        self.progress_bar.setValue(progress)
        
    def update_log(self, message):
        """
        更新日志显示
        
        参数:
            message: 日志消息
        """
        self.log_text.append(message)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_log(self):
        """
        清空日志
        """
        self.log_text.clear()
        
    def closeEvent(self, event):
        """
        关闭事件处理
        确保线程正确结束
        
        参数:
            event: 关闭事件
        """
        if self.inference_thread and self.inference_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认关闭",
                "正在处理中，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# 便捷函数
def show_nanodet_inference_dialog(parent=None):
    """
    显示NanoDet推理对话框的便捷函数
    
    参数:
        parent: 父窗口
        
    返回:
        dialog: 对话框实例
    """
    dialog = NanoDetInferenceDialog(parent)
    dialog.show()
    return dialog


if __name__ == "__main__":
    # 测试代码
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = NanoDetInferenceDialog()
    dialog.show()
    sys.exit(app.exec_())