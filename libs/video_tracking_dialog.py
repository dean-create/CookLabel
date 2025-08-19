#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频目标追踪取图配置对话框
用于配置视频路径、YOLO模型和帧采样间隔等参数
"""

import os
import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QSpinBox, QComboBox,
                             QGroupBox, QFormLayout, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class VideoTrackingDialog(QDialog):
    """视频目标追踪配置对话框"""
    
    # 定义信号，用于传递配置参数
    config_confirmed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super(VideoTrackingDialog, self).__init__(parent)
        self.setWindowTitle("目标追踪取图配置")
        
        # 设置窗口标志，包含最小化、最大化和关闭按钮
        self.setWindowFlags(Qt.Dialog | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        
        # 设置窗口大小和位置
        self.resize(700, 550)  # 使用resize而不是setFixedSize，允许用户调整大小
        self.center_on_screen()  # 居中显示
        
        # 初始化配置参数
        self.config = {
            'video_path': '',
            'model_type': 'yolov8n.pt',  # 默认使用官方预训练模型
            'custom_model_path': '',
            'frame_interval': 5,  # 默认每5帧取一张图
            'output_dir': ''
        }
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        layout.setSpacing(15)  # 增加组件间距，提高布局美观性
        layout.setContentsMargins(20, 20, 20, 20)  # 增加边距
        
        # 设置标题字体 - 增大字体提高可读性
        title_font = QFont()
        title_font.setPointSize(14)  # 从12增加到14
        title_font.setBold(True)
        
        # 设置内容字体
        content_font = QFont()
        content_font.setPointSize(11)  # 设置内容字体大小
        
        # 视频文件选择组
        video_group = QGroupBox("视频文件选择")
        video_group.setFont(title_font)
        video_layout = QFormLayout()
        video_layout.setSpacing(10)  # 增加表单内部间距
        
        # 视频路径选择
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("请选择视频文件...")
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setFont(content_font)  # 设置内容字体
        self.video_path_edit.setMinimumHeight(35)  # 增加输入框高度
        
        video_browse_btn = QPushButton("浏览")
        video_browse_btn.clicked.connect(self.browse_video_file)
        video_browse_btn.setFont(content_font)  # 设置按钮字体
        video_browse_btn.setMinimumHeight(35)  # 增加按钮高度
        video_browse_btn.setMinimumWidth(80)  # 设置按钮最小宽度
        
        video_path_layout = QHBoxLayout()
        video_path_layout.addWidget(self.video_path_edit)
        video_path_layout.addWidget(video_browse_btn)
        
        video_layout.addRow("视频文件:", video_path_layout)
        video_group.setLayout(video_layout)
        
        # YOLO模型选择组
        model_group = QGroupBox("YOLO模型选择")
        model_group.setFont(title_font)
        model_layout = QFormLayout()
        model_layout.setSpacing(10)  # 增加表单内部间距
        
        # 模型类型选择
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n.pt (官方预训练-轻量)",
            "yolov8s.pt (官方预训练-小型)",
            "yolov8m.pt (官方预训练-中型)",
            "yolov8l.pt (官方预训练-大型)",
            "yolov8x.pt (官方预训练-超大)",
            "自定义模型"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_type_changed)
        self.model_combo.setFont(content_font)  # 设置下拉框字体
        self.model_combo.setMinimumHeight(35)  # 增加下拉框高度
        
        # 自定义模型路径选择（默认隐藏）
        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("请选择自定义模型文件...")
        self.custom_model_edit.setReadOnly(True)
        self.custom_model_edit.setVisible(False)
        self.custom_model_edit.setFont(content_font)  # 设置字体
        self.custom_model_edit.setMinimumHeight(35)  # 增加高度
        
        self.custom_model_btn = QPushButton("浏览")
        self.custom_model_btn.clicked.connect(self.browse_custom_model)
        self.custom_model_btn.setVisible(False)
        self.custom_model_btn.setFont(content_font)  # 设置按钮字体
        self.custom_model_btn.setMinimumHeight(35)  # 增加按钮高度
        self.custom_model_btn.setMinimumWidth(80)  # 设置按钮最小宽度
        
        custom_model_layout = QHBoxLayout()
        custom_model_layout.addWidget(self.custom_model_edit)
        custom_model_layout.addWidget(self.custom_model_btn)
        
        model_layout.addRow("模型类型:", self.model_combo)
        model_layout.addRow("自定义模型:", custom_model_layout)
        model_group.setLayout(model_layout)
        
        # 采样参数组
        sampling_group = QGroupBox("采样参数")
        sampling_group.setFont(title_font)
        sampling_layout = QFormLayout()
        sampling_layout.setSpacing(10)  # 增加表单内部间距
        
        # 帧间隔设置
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 100)
        self.frame_interval_spin.setValue(5)
        self.frame_interval_spin.setSuffix(" 帧")
        self.frame_interval_spin.setToolTip("设置每隔多少帧截取一张图片")
        self.frame_interval_spin.setFont(content_font)  # 设置字体
        self.frame_interval_spin.setMinimumHeight(35)  # 增加高度
        self.frame_interval_spin.setMinimumWidth(120)  # 设置最小宽度
        
        sampling_layout.addRow("帧间隔:", self.frame_interval_spin)
        sampling_group.setLayout(sampling_layout)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        self.confirm_btn = QPushButton("确认配置")
        self.confirm_btn.clicked.connect(self.confirm_config)
        self.confirm_btn.setFont(content_font)  # 设置按钮字体
        self.confirm_btn.setMinimumHeight(40)  # 增加按钮高度
        self.confirm_btn.setMinimumWidth(120)  # 设置按钮最小宽度
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setFont(content_font)  # 设置按钮字体
        self.cancel_btn.setMinimumHeight(40)  # 增加按钮高度
        self.cancel_btn.setMinimumWidth(120)  # 设置按钮最小宽度
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 25px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.confirm_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # 添加所有组件到主布局
        layout.addWidget(video_group)
        layout.addWidget(model_group)
        layout.addWidget(sampling_group)
        layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def center_on_screen(self):
        """将窗口居中显示在屏幕上"""
        from PyQt5.QtWidgets import QDesktopWidget
        
        # 获取屏幕几何信息
        screen = QDesktopWidget().screenGeometry()
        
        # 获取窗口几何信息
        window = self.geometry()
        
        # 计算居中位置
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        
        # 移动窗口到居中位置
        self.move(x, y)
        
    def browse_video_file(self):
        """浏览选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;所有文件 (*)"
        )
        
        if file_path:
            self.video_path_edit.setText(file_path)
            self.config['video_path'] = file_path
            
            # 自动设置输出目录为视频文件同级目录
            video_dir = os.path.dirname(file_path)
            self.config['output_dir'] = os.path.join(video_dir, "Auto_dataset")
            
    def on_model_type_changed(self, text):
        """模型类型改变时的处理"""
        if "自定义模型" in text:
            # 显示自定义模型选择控件
            self.custom_model_edit.setVisible(True)
            self.custom_model_btn.setVisible(True)
            self.config['model_type'] = 'custom'
        else:
            # 隐藏自定义模型选择控件
            self.custom_model_edit.setVisible(False)
            self.custom_model_btn.setVisible(False)
            
            # 提取模型名称
            model_name = text.split(' ')[0]  # 例如从 "yolov8n.pt (官方预训练-轻量)" 提取 "yolov8n.pt"
            self.config['model_type'] = model_name
            
    def browse_custom_model(self):
        """浏览选择自定义模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            "",
            "模型文件 (*.pt *.pth *.onnx);;所有文件 (*)"
        )
        
        if file_path:
            self.custom_model_edit.setText(file_path)
            self.config['custom_model_path'] = file_path
            
    def confirm_config(self):
        """确认配置"""
        # 验证必要参数
        if not self.config['video_path']:
            QMessageBox.warning(self, "配置错误", "请选择视频文件！")
            return
            
        if not os.path.exists(self.config['video_path']):
            QMessageBox.warning(self, "配置错误", "选择的视频文件不存在！")
            return
            
        # 如果选择自定义模型，验证模型文件
        if self.config['model_type'] == 'custom':
            if not self.config['custom_model_path']:
                QMessageBox.warning(self, "配置错误", "请选择自定义模型文件！")
                return
            if not os.path.exists(self.config['custom_model_path']):
                QMessageBox.warning(self, "配置错误", "选择的模型文件不存在！")
                return
                
        # 更新帧间隔配置
        self.config['frame_interval'] = self.frame_interval_spin.value()
        
        # 直接启动视频追踪界面，实现流畅切换
        self.start_tracking_interface()
        
    def start_tracking_interface(self):
        """启动视频追踪界面"""
        try:
            # 导入视频追踪界面
            from libs.video_tracking_interface import VideoTrackingInterface
            
            # 创建视频追踪工具主界面 - 使用全局变量保持引用，防止被垃圾回收
            global tracking_interface_instance
            tracking_interface_instance = VideoTrackingInterface(self.config.copy())
            
            # 设置窗口属性，确保独立显示
            tracking_interface_instance.setWindowTitle("视频目标追踪取图工具")
            
            # 设置窗口为独立窗口，不依赖父窗口
            tracking_interface_instance.setParent(None)
            tracking_interface_instance.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
            
            # 显示视频追踪工具窗口
            tracking_interface_instance.show()
            
            # 确保窗口显示在前台并获得焦点
            tracking_interface_instance.raise_()
            tracking_interface_instance.activateWindow()
            
            # 自动加载视频
            if self.config.get('video_path'):
                tracking_interface_instance.load_video(self.config['video_path'])
            
            # 关闭配置对话框，实现流畅切换
            self.accept()
            
        except ImportError as e:
            QMessageBox.critical(self, "导入错误", f"无法导入视频追踪模块:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动视频追踪工具失败:\n{str(e)}")
        
    def get_config(self):
        """获取当前配置"""
        return self.config.copy()


if __name__ == "__main__":
    """测试代码"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    def on_config_confirmed(config):
        print("配置确认:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    dialog = VideoTrackingDialog()
    dialog.config_confirmed.connect(on_config_confirmed)
    dialog.show()
    
    sys.exit(app.exec_())