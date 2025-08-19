#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
类别定义对话框
用于YOLO格式标签的类别定义
"""

import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QSpinBox, QLineEdit, QTextEdit,
                            QMessageBox, QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ClassDefinitionDialog(QDialog):
    """类别定义对话框"""
    
    def __init__(self, classes_file_path, parent=None):
        super(ClassDefinitionDialog, self).__init__(parent)
        self.classes_file_path = classes_file_path
        self.class_inputs = []  # 存储类别输入框
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("YOLO类别定义")
        self.setModal(True)
        self.resize(500, 600)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题和说明
        title_label = QLabel("YOLO类别定义")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 说明文本
        info_label = QLabel(
            "检测到YOLO格式的标签文件，但缺少classes.txt文件。\n"
            "程序已为您创建了一个空的classes.txt文件。\n"
            "请定义您需要的类别数量和具体类别名称："
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        main_layout.addWidget(info_label)
        
        # 类别数量选择
        count_layout = QHBoxLayout()
        count_label = QLabel("类别数量:")
        count_label.setMinimumWidth(80)
        self.count_spinbox = QSpinBox()
        self.count_spinbox.setMinimum(1)
        self.count_spinbox.setMaximum(100)
        self.count_spinbox.setValue(1)
        self.count_spinbox.valueChanged.connect(self.on_count_changed)
        
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.count_spinbox)
        count_layout.addStretch()
        main_layout.addLayout(count_layout)
        
        # 滚动区域用于显示类别输入框
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(scroll_area)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 确定按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.setMinimumHeight(35)
        self.ok_button.setStyleSheet("""
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
        """)
        self.ok_button.clicked.connect(self.accept_classes)
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.setMinimumHeight(35)
        cancel_button.setStyleSheet("""
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
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)
        
        # 初始化类别输入框
        self.on_count_changed()
        
    def on_count_changed(self):
        """当类别数量改变时更新输入框"""
        # 清除现有的输入框
        for i in reversed(range(self.scroll_layout.count())):
            child = self.scroll_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.class_inputs.clear()
        
        # 创建新的输入框
        count = self.count_spinbox.value()
        for i in range(count):
            class_layout = QHBoxLayout()
            
            # 类别标签
            label = QLabel(f"类别 {i}:")
            label.setMinimumWidth(60)
            
            # 类别输入框
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"请输入第{i}个类别名称")
            input_field.setMinimumHeight(30)
            
            # 设置默认值
            if i == 0:
                input_field.setText("person")
            elif i == 1:
                input_field.setText("car")
            elif i == 2:
                input_field.setText("bicycle")
            
            self.class_inputs.append(input_field)
            
            class_layout.addWidget(label)
            class_layout.addWidget(input_field)
            
            # 创建容器widget
            container = QWidget()
            container.setLayout(class_layout)
            self.scroll_layout.addWidget(container)
        
        # 添加弹性空间
        self.scroll_layout.addStretch()
        
    def accept_classes(self):
        """确认并保存类别定义"""
        # 获取所有类别名称
        class_names = []
        for input_field in self.class_inputs:
            class_name = input_field.text().strip()
            if not class_name:
                QMessageBox.warning(self, "警告", "请填写所有类别名称！")
                return
            if class_name in class_names:
                QMessageBox.warning(self, "警告", f"类别名称 '{class_name}' 重复，请使用不同的名称！")
                return
            class_names.append(class_name)
        
        # 保存到classes.txt文件
        try:
            with open(self.classes_file_path, 'w', encoding='utf-8') as f:
                for class_name in class_names:
                    f.write(class_name + '\n')
            
            QMessageBox.information(self, "成功", 
                                  f"类别定义已保存到:\n{self.classes_file_path}\n\n"
                                  f"共定义了 {len(class_names)} 个类别:\n" + 
                                  "\n".join([f"{i}: {name}" for i, name in enumerate(class_names)]))
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存classes.txt文件失败:\n{str(e)}")