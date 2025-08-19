#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
from xml.etree import ElementTree
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QFileDialog, QMessageBox, 
                           QProgressBar, QTextEdit, QGroupBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


def imread_chinese(img_path):
    """
    支持中文路径的图片读取函数
    使用cv2.imdecode配合numpy来读取包含中文字符的路径
    
    参数:
        img_path: 图片文件路径（可包含中文字符）
    
    返回:
        numpy数组格式的图片数据，如果读取失败返回None
    """
    try:
        # 使用numpy读取文件的二进制数据
        with open(img_path, 'rb') as f:
            img_data = f.read()
        
        # 将二进制数据转换为numpy数组
        img_array = np.frombuffer(img_data, np.uint8)
        
        # 使用cv2.imdecode解码图片数据
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"读取图片失败: {img_path}, 错误: {e}")
        return None


class CropWorker(QThread):
    """裁剪工作线程"""
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # 成功数量, 总数量
    
    def __init__(self, txt_folder, img_folder, save_folder, roi_ratio):
        super().__init__()
        self.txt_folder = txt_folder
        self.img_folder = img_folder
        self.save_folder = save_folder
        self.roi_ratio = roi_ratio
        self.is_cancelled = False
    
    def cancel(self):
        self.is_cancelled = True
    
    def find_corresponding_label_file(self, img_path, label_folder):
        """查找与图片对应的标注文件（支持TXT和XML格式）
        要求图片名称与标签名称完全一致才匹配
        """
        # 获取图片文件名（不含扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 优先查找TXT文件，然后查找XML文件
        for ext in ['.txt', '.xml']:
            # 在标注文件夹中递归查找对应的标注文件
            for root, _, files in os.walk(label_folder):
                for file in files:
                    if file.endswith(ext):
                        label_name = os.path.splitext(file)[0]
                        # 只有完全匹配的文件名才认为是对应的标注文件
                        if label_name == img_name:
                            return os.path.join(root, file), ext
        
        return None, None
    
    def parse_xml_annotation(self, xml_path):
        """解析XML格式的标注文件，返回第一个边界框的坐标信息"""
        try:
            # 解析XML文件
            tree = ElementTree.parse(xml_path)
            root = tree.getroot()
            
            # 获取图片尺寸信息
            size_elem = root.find('size')
            if size_elem is None:
                return None
            
            img_width = int(size_elem.find('width').text)
            img_height = int(size_elem.find('height').text)
            
            # 查找第一个object元素
            object_elem = root.find('object')
            if object_elem is None:
                return None
            
            # 获取边界框坐标
            bndbox = object_elem.find('bndbox')
            if bndbox is None:
                return None
            
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # 转换为YOLO格式的归一化坐标（中心点坐标和宽高）
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            box_width = (xmax - xmin) / img_width
            box_height = (ymax - ymin) / img_height
            
            return x_center, y_center, box_width, box_height
            
        except Exception as e:
            print(f"解析XML文件失败: {xml_path}, 错误: {e}")
            return None
    
    def parse_txt_annotation(self, txt_path):
        """解析TXT格式的YOLO标注文件，返回第一行的坐标信息
        如果文件为空或格式不正确，返回None
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                # 读取文件内容并去除空白行
                content = f.read().strip()
                
                # 检查文件是否为空
                if not content:
                    return None
                
                # 获取第一行有效内容
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                if not lines:
                    return None
                
                first_line = lines[0]
                parts = first_line.split()
                
                # 检查格式是否正确（应该有5个数值：class_id x_center y_center width height）
                if len(parts) != 5:
                    return None
                
                # 尝试转换为浮点数
                try:
                    _, x_center, y_center, box_w, box_h = map(float, parts)
                    
                    # 验证坐标值是否在合理范围内（YOLO格式应该是0-1之间的归一化坐标）
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 <= box_w <= 1 and 0 <= box_h <= 1):
                        return None
                    
                    return x_center, y_center, box_w, box_h
                    
                except ValueError:
                    # 数值转换失败
                    return None
                
        except Exception as e:
            print(f"解析TXT文件失败: {txt_path}, 错误: {e}")
            return None
    
    def run(self):
        """执行裁剪任务"""
        try:
            # 统计图片数量
            total_images = 0
            image_files = []
            
            for root, _, files in os.walk(self.img_folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_files.append(os.path.join(root, file))
                        total_images += 1
            
            if total_images == 0:
                self.log_updated.emit("未找到任何图片文件！")
                self.finished.emit(0, 0)
                return
            
            self.log_updated.emit(f"找到 {total_images} 张图片，开始裁剪...")
            self.log_updated.emit("支持的标注格式：YOLO(.txt) 和 XML(.xml)")
            
            # 确保保存文件夹存在
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
                self.log_updated.emit(f"创建保存文件夹: {self.save_folder}")
            
            processed_count = 0
            crop_count = 0
            skipped_count = 0  # 跳过的图片计数
            
            for i, img_path in enumerate(image_files):
                if self.is_cancelled:
                    break
                
                img_name = os.path.basename(img_path)
                
                # 查找对应的标注文件（TXT或XML格式）
                label_path, label_ext = self.find_corresponding_label_file(img_path, self.txt_folder)
                
                if label_path and os.path.exists(label_path):
                    try:
                        # 根据文件类型解析标注文件
                        if label_ext == '.txt':
                            # 解析TXT格式的YOLO标注文件
                            coords = self.parse_txt_annotation(label_path)
                            if coords is None:
                                # 检查文件是否为空
                                try:
                                    with open(label_path, 'r', encoding='utf-8') as f:
                                        content = f.read().strip()
                                        if not content:
                                            self.log_updated.emit(f"跳过：{img_name} - TXT文件为空")
                                        else:
                                            self.log_updated.emit(f"跳过：{img_name} - TXT文件格式不正确或坐标值无效")
                                except:
                                    self.log_updated.emit(f"跳过：{img_name} - 无法读取TXT文件")
                                
                                processed_count += 1
                                skipped_count += 1
                                continue
                            x_center, y_center, box_w, box_h = coords
                            
                        elif label_ext == '.xml':
                            # 解析XML格式的标注文件
                            coords = self.parse_xml_annotation(label_path)
                            if coords is None:
                                self.log_updated.emit(f"跳过：{img_name} - XML文件格式不正确或无边界框")
                                processed_count += 1
                                skipped_count += 1
                                continue
                            x_center, y_center, box_w, box_h = coords
                            
                        else:
                            self.log_updated.emit(f"跳过：{img_name} - 不支持的标注文件格式 {label_ext}")
                            processed_count += 1
                            skipped_count += 1
                            continue
                        
                        # 读取图片并裁剪（支持中文路径）
                        img = imread_chinese(img_path)
                        if img is not None:
                            h, w, _ = img.shape
                            
                            # 计算实际像素坐标
                            x_center_pixel = int(x_center * w)
                            y_center_pixel = int(y_center * h)
                            box_w_pixel = int(box_w * w * self.roi_ratio)
                            box_h_pixel = int(box_h * h * self.roi_ratio)
                            
                            # 计算裁剪区域
                            x1 = max(0, x_center_pixel - box_w_pixel // 2)
                            y1 = max(0, y_center_pixel - box_h_pixel // 2)
                            x2 = min(w, x_center_pixel + box_w_pixel // 2)
                            y2 = min(h, y_center_pixel + box_h_pixel // 2)
                            
                            # 裁剪图片
                            cropped_img = img[y1:y2, x1:x2]
                            
                            # 保存裁剪后的图片
                            img_name = os.path.basename(img_path)
                            save_path = os.path.join(self.save_folder, img_name)
                            
                            # 使用cv2.imencode支持中文路径保存
                            _, encoded_img = cv2.imencode(os.path.splitext(img_name)[1], cropped_img)
                            with open(save_path, 'wb') as f:
                                f.write(encoded_img.tobytes())
                            
                            crop_count += 1
                            # 显示使用的标注文件格式
                            format_name = "YOLO" if label_ext == '.txt' else "XML"
                            self.log_updated.emit(f"裁剪完成: {img_name} (使用{format_name}格式)")
                        else:
                            self.log_updated.emit(f"无法读取图片: {os.path.basename(img_path)}")
                    
                    except Exception as e:
                        self.log_updated.emit(f"跳过：{img_name} - 处理时出错: {str(e)}")
                        skipped_count += 1
                else:
                    self.log_updated.emit(f"跳过：{img_name} - 未找到对应的标注文件（要求文件名完全一致）")
                    skipped_count += 1
                
                processed_count += 1
                progress = int((processed_count / total_images) * 100)
                self.progress_updated.emit(progress)
            
            # 输出最终统计信息
            self.log_updated.emit(f"裁剪完成！成功: {crop_count}张, 跳过: {skipped_count}张, 总计: {total_images}张")
            self.finished.emit(crop_count, total_images)
            
        except Exception as e:
            self.log_updated.emit(f"裁剪过程中发生错误: {str(e)}")
            self.finished.emit(0, 0)


class CropDialog(QDialog):
    """图片裁剪对话框"""
    
    def __init__(self, current_image_path=None, label_dir=None, parent=None):
        super().__init__(parent)
        self.current_image_path = current_image_path
        self.label_dir = label_dir  # 标签文件夹路径
        self.crop_worker = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('图片裁剪工具 (支持YOLO和XML格式)')
        self.setFixedSize(1100, 600)  # 大幅增加窗口尺寸，提供更好的用户体验
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # 裁剪参数组
        param_group = QGroupBox("裁剪参数")
        param_layout = QVBoxLayout()
        
        # ROI比例设置
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("裁剪区域缩放比例:"))
        self.roi_spinbox = QDoubleSpinBox()
        self.roi_spinbox.setRange(0.1, 5.0)
        self.roi_spinbox.setSingleStep(0.1)
        self.roi_spinbox.setValue(1.0)
        self.roi_spinbox.setDecimals(1)
        self.roi_spinbox.valueChanged.connect(self.on_roi_changed)  # 连接值改变信号
        roi_layout.addWidget(self.roi_spinbox)
        roi_layout.addStretch()
        param_layout.addLayout(roi_layout)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 日志区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(300)  # 设置日志区域最小高度，确保用户能清楚看到日志内容
        self.log_area.setFont(QFont("Consolas", 10))  # 稍微增大字体，提高可读性
        log_layout.addWidget(self.log_area)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始裁剪")
        self.start_button.clicked.connect(self.start_crop)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 自动设置所有路径
        self.auto_setup_paths()
    
    def auto_setup_paths(self):
        """自动设置所有路径"""
        if not self.current_image_path:
            self.start_button.setEnabled(False)
            return
        
        # 自动推导图片文件夹
        img_dir = os.path.dirname(self.current_image_path)
        
        # 使用传递的标签文件夹路径，如果没有则使用图片文件夹
        txt_dir = self.label_dir if self.label_dir else img_dir
        
        # 自动设置保存文件夹，修复浮点数精度问题
        parent_dir = os.path.dirname(img_dir)
        roi_ratio = self.roi_spinbox.value()
        if roi_ratio == 1.0:
            save_folder = os.path.join(parent_dir, "resize")
        else:
            # 格式化浮点数，去除多余的零和小数点
            ratio_str = f"{roi_ratio:.1f}".rstrip('0').rstrip('.')
            save_folder = os.path.join(parent_dir, f"resize_{ratio_str}")
        
        # 保存路径到内部变量（用于后续处理）
        self.txt_path_value = txt_dir
        self.img_path_value = img_dir
        self.save_path_value = save_folder
        
        # 检查是否可以开始裁剪
        self.check_ready()
    
    def on_roi_changed(self):
        """ROI比例改变时的处理"""
        if self.current_image_path:
            self.auto_setup_paths()
    
    def check_ready(self):
        """检查是否可以开始裁剪"""
        ready = bool(hasattr(self, 'txt_path_value') and 
                    hasattr(self, 'img_path_value') and 
                    hasattr(self, 'save_path_value') and
                    self.txt_path_value and 
                    self.img_path_value and 
                    self.save_path_value)
        self.start_button.setEnabled(ready)
    
    def start_crop(self):
        """开始裁剪"""
        txt_folder = self.txt_path_value
        img_folder = self.img_path_value
        save_folder = self.save_path_value
        roi_ratio = self.roi_spinbox.value()
        
        if not all([txt_folder, img_folder, save_folder]):
            QMessageBox.warning(self, "警告", "路径设置有误，请重新打开对话框！")
            return
        
        # 禁用开始按钮，启用取消按钮
        self.start_button.setEnabled(False)
        self.cancel_button.setText("停止")
        
        # 清空日志
        self.log_area.clear()
        self.progress_bar.setValue(0)
        
        # 添加开始日志
        self.add_log(f"开始裁剪，缩放比例: {roi_ratio}")
        self.add_log(f"图片文件夹: {img_folder}")
        self.add_log(f"标签文件夹: {txt_folder}")
        self.add_log(f"保存到: {save_folder}")
        self.add_log("=" * 50)
        
        # 创建并启动工作线程
        self.crop_worker = CropWorker(txt_folder, img_folder, save_folder, roi_ratio)
        self.crop_worker.progress_updated.connect(self.progress_bar.setValue)
        self.crop_worker.log_updated.connect(self.add_log)
        self.crop_worker.finished.connect(self.crop_finished)
        self.crop_worker.start()
    
    def add_log(self, message):
        """添加日志"""
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    
    def crop_finished(self, success_count, total_count):
        """裁剪完成"""
        self.add_log(f"裁剪完成！成功裁剪 {success_count}/{total_count} 张图片")
        self.progress_bar.setValue(100)
        
        # 恢复按钮状态
        self.start_button.setEnabled(True)
        self.cancel_button.setText("关闭")
        
        if success_count > 0:
            QMessageBox.information(self, "完成", f"裁剪完成！\n成功裁剪 {success_count}/{total_count} 张图片")
    
    def reject(self):
        """取消或关闭对话框"""
        if self.crop_worker and self.crop_worker.isRunning():
            self.crop_worker.cancel()
            self.crop_worker.wait()
        super().reject()