#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频拆帧工具 - Qt界面版本
支持用户图形化选择视频文件夹、输出文件夹，并自定义抽帧间隔
"""

import os
import cv2
from PIL import Image
import re
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

from libs.utils import new_icon


class VideoFrameExtractorDialog(QDialog):
    """视频拆帧工具对话框"""
    
    # 定义信号，用于线程间通信
    if pyqt_version == 5:
        progress_updated = pyqtSignal(int)  # 进度更新信号
        log_updated = pyqtSignal(str)      # 日志更新信号
        extraction_finished = pyqtSignal(dict)  # 拆帧完成信号
        frame_progress_updated = pyqtSignal(int, int, str)  # 帧级别进度更新信号 (当前帧, 总帧数, 视频名)
    else:
        progress_updated = pyqtSignal(int)  # 进度更新信号
        log_updated = pyqtSignal(str)      # 日志更新信号
        extraction_finished = pyqtSignal(dict)  # 拆帧完成信号
        frame_progress_updated = pyqtSignal(int, int, str)  # 帧级别进度更新信号
    
    def __init__(self, parent=None):
        super(VideoFrameExtractorDialog, self).__init__(parent)
        self.setWindowTitle("视频拆帧工具")
        self.setWindowIcon(new_icon('app'))
        self.setMinimumSize(900, 650)  # 增大最小尺寸
        self.resize(1000, 750)         # 增大默认尺寸
        
        # 初始化变量
        self.video_folder_path = ""
        self.output_folder_path = ""
        self.frame_interval = 15  # 默认每隔15帧抽取一张
        self.max_workers = 4      # 默认并发数
        self.is_extracting = False  # 是否正在拆帧
        self.extraction_thread = None  # 拆帧线程
        
        # 初始化界面
        self.init_ui()
        
        # 连接信号
        self.progress_updated.connect(self.update_progress)
        self.log_updated.connect(self.append_log)
        self.extraction_finished.connect(self.on_extraction_finished)
        self.frame_progress_updated.connect(self.update_frame_progress)
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("视频拆帧工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 路径选择区域
        path_group = QGroupBox("路径设置")
        path_layout = QVBoxLayout()
        
        # 视频文件夹选择
        video_folder_label_title = QLabel("视频文件夹:")
        video_folder_label_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        path_layout.addWidget(video_folder_label_title)
        
        video_folder_layout = QHBoxLayout()
        self.video_folder_label = QLabel("请选择视频文件夹...")
        self.video_folder_label.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        self.video_folder_label.setWordWrap(True)  # 允许文本换行
        self.video_folder_btn = QPushButton("浏览")
        self.video_folder_btn.setFixedWidth(80)  # 固定按钮宽度
        self.video_folder_btn.clicked.connect(self.select_video_folder)
        video_folder_layout.addWidget(self.video_folder_label, 1)
        video_folder_layout.addWidget(self.video_folder_btn)
        path_layout.addLayout(video_folder_layout)
        
        # 添加间距
        path_layout.addSpacing(10)
        
        # 输出文件夹选择
        output_folder_label_title = QLabel("输出文件夹:")
        output_folder_label_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        path_layout.addWidget(output_folder_label_title)
        
        output_folder_layout = QHBoxLayout()
        self.output_folder_label = QLabel("请选择输出文件夹...")
        self.output_folder_label.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        self.output_folder_label.setWordWrap(True)  # 允许文本换行
        self.output_folder_btn = QPushButton("浏览")
        self.output_folder_btn.setFixedWidth(80)  # 固定按钮宽度
        self.output_folder_btn.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(self.output_folder_label, 1)
        output_folder_layout.addWidget(self.output_folder_btn)
        path_layout.addLayout(output_folder_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 参数设置区域
        params_group = QGroupBox("参数设置")
        params_layout = QHBoxLayout()
        params_layout.setSpacing(30)  # 增加控件间距
        
        # 抽帧间隔
        interval_layout = QVBoxLayout()
        interval_label = QLabel("抽帧间隔:")
        interval_label.setStyleSheet("font-weight: bold;")
        interval_layout.addWidget(interval_label)
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1, 1000)
        self.interval_spinbox.setValue(15)
        self.interval_spinbox.setSuffix(" 帧")
        self.interval_spinbox.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        interval_layout.addWidget(self.interval_spinbox)
        params_layout.addLayout(interval_layout)
        
        # 并发数
        workers_layout = QVBoxLayout()
        workers_label = QLabel("并发数:")
        workers_label.setStyleSheet("font-weight: bold;")
        workers_layout.addWidget(workers_label)
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setRange(1, 16)
        self.workers_spinbox.setValue(4)
        self.workers_spinbox.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        workers_layout.addWidget(self.workers_spinbox)
        params_layout.addLayout(workers_layout)
        
        params_layout.addStretch()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 进度显示区域
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                background-color: #f8f8f8;
                padding: 8px;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("处理失败或被中断")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            border: 1px solid #ccc; 
            padding: 8px; 
            background-color: #f8f8f8;
            border-radius: 4px;
            min-height: 20px;
        """)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 处理日志
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        # 增大日志字体
        font = self.log_text.font()
        font.setPointSize(11)  # 增大字体到11pt
        self.log_text.setFont(font)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                background-color: #ffffff;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11pt;
                line-height: 1.4;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("开始拆帧")
        self.start_btn.setIcon(new_icon('play'))
        self.start_btn.setMinimumSize(100, 35)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_extraction)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止拆帧")
        self.stop_btn.setIcon(new_icon('stop'))
        self.stop_btn.setMinimumSize(100, 35)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_extraction)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addSpacing(20)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.setMinimumSize(80, 35)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def select_video_folder(self):
        """选择视频文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if folder:
            self.video_folder_path = folder
            self.video_folder_label.setText(folder)
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_path = folder
            self.output_folder_label.setText(folder)
    
    def get_video_files(self, folder_path):
        """获取文件夹中的所有视频文件"""
        video_extensions = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        video_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return video_files
    
    def start_extraction(self):
        """开始拆帧处理"""
        # 验证输入
        if not self.video_folder_path:
            QMessageBox.warning(self, "警告", "请先选择视频文件夹！")
            return
        
        if not self.output_folder_path:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return
        
        # 获取参数
        self.frame_interval = self.interval_spinbox.value()
        self.max_workers = self.workers_spinbox.value()
        
        # 创建输出文件夹
        output_images_folder = os.path.join(self.output_folder_path, "output_images")
        os.makedirs(output_images_folder, exist_ok=True)
        
        # 更新界面状态
        self.is_extracting = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在处理...")
        self.log_text.clear()
        
        # 获取视频文件列表
        video_files = self.get_video_files(self.video_folder_path)
        self.append_log(f"找到 {len(video_files)} 个视频文件")
        self.append_log(f"抽帧间隔: {self.frame_interval} 帧")
        self.append_log(f"并发数: {self.max_workers}")
        self.append_log(f"输出文件夹: {output_images_folder}")
        self.append_log("开始处理...")
        
        # 启动拆帧线程
        self.extraction_thread = ExtractionThread(
            video_files, 
            output_images_folder, 
            self.frame_interval, 
            self.max_workers,
            self
        )
        self.extraction_thread.start()
    
    def stop_extraction(self):
        """停止拆帧处理"""
        if self.extraction_thread and self.extraction_thread.isRunning():
            self.extraction_thread.stop()
            self.append_log("正在停止处理...")
            self.status_label.setText("正在停止...")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_frame_progress(self, current_frame, total_frames, video_name):
        """更新帧级别进度"""
        if total_frames > 0:
            frame_progress = int((current_frame / total_frames) * 100)
            self.append_log(f"正在处理 {video_name}: 第 {current_frame}/{total_frames} 帧 ({frame_progress}%)")
    
    def append_log(self, message):
        """添加日志信息"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_extraction_finished(self, results):
        """拆帧完成处理"""
        self.is_extracting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if results.get('success', False):
            self.progress_bar.setValue(100)
            self.status_label.setText("处理完成")
            
            # 显示统计信息
            stats = results.get('stats', {})
            self.append_log("=" * 50)
            self.append_log("拆帧处理完成！")
            self.append_log(f"处理总耗时: {stats.get('elapsed_time', '未知')}")
            self.append_log(f"处理视频数量: {stats.get('video_count', 0)}")
            self.append_log(f"生成图片数量: {stats.get('image_count', 0)}")
            self.append_log("=" * 50)
            
            # 显示完成对话框
            QMessageBox.information(
                self, 
                "完成", 
                f"拆帧处理完成！\n\n"
                f"处理视频数量: {stats.get('video_count', 0)}\n"
                f"生成图片数量: {stats.get('image_count', 0)}\n"
                f"处理耗时: {stats.get('elapsed_time', '未知')}"
            )
        else:
            self.status_label.setText("处理失败或被中断")
            error_msg = results.get('error', '未知错误')
            self.append_log(f"处理失败: {error_msg}")
            QMessageBox.warning(self, "错误", f"拆帧处理失败:\n{error_msg}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.is_extracting:
            reply = QMessageBox.question(
                self, 
                "确认关闭", 
                "拆帧处理正在进行中，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_extraction()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class ExtractionThread(QThread):
    """拆帧处理线程"""
    
    def __init__(self, video_files, output_folder, frame_interval, max_workers, parent_dialog):
        super(ExtractionThread, self).__init__()
        self.video_files = video_files
        self.output_folder = output_folder
        self.frame_interval = frame_interval
        self.max_workers = max_workers
        self.parent_dialog = parent_dialog
        self.should_stop = False
        self.total_frames_all_videos = 0
        self.processed_frames_all_videos = 0
    
    def stop(self):
        """停止线程"""
        self.should_stop = True
    
    def run(self):
        """线程主函数"""
        try:
            start_time = time.time()
            
            # 首先计算所有视频的总帧数
            self.parent_dialog.log_updated.emit("正在计算总帧数...")
            self.calculate_total_frames()
            
            # 使用简单的顺序处理方式
            processed_videos = []
            image_counter = 0
            results = {}
            
            total_videos = len(self.video_files)
            
            # 添加处理开始的统计信息
            self.parent_dialog.log_updated.emit("=" * 60)
            self.parent_dialog.log_updated.emit("🎬 开始视频拆帧处理")
            self.parent_dialog.log_updated.emit(f"📁 待处理视频数量: {total_videos}")
            self.parent_dialog.log_updated.emit(f"🎯 抽帧间隔: 每 {self.frame_interval} 帧抽取一张")
            self.parent_dialog.log_updated.emit(f"📊 预计总帧数: {self.total_frames_all_videos}")
            self.parent_dialog.log_updated.emit("=" * 60)
            
            for i, video_file in enumerate(self.video_files):
                if self.should_stop:
                    self.parent_dialog.extraction_finished.emit({
                        'success': False,
                        'error': '用户中断处理'
                    })
                    return
                
                # 处理单个视频
                saved_count = self.process_video_with_progress(video_file, processed_videos, i + 1, total_videos)
                results[video_file] = saved_count
                image_counter += saved_count
                
                # 实时显示累计图片数量
                self.parent_dialog.log_updated.emit(f"📈 当前累计生成图片: {image_counter} 张")
            
            if self.should_stop:
                self.parent_dialog.extraction_finished.emit({
                    'success': False,
                    'error': '用户中断处理'
                })
                return
            
            # 计算统计信息
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))
            
            # 获取实际输出文件夹中的图片计数
            actual_count = len(glob.glob(os.path.join(self.output_folder, "*.jpg")))
            
            # 在日志中显示详细的图片统计信息
            self.parent_dialog.log_updated.emit("=" * 60)
            self.parent_dialog.log_updated.emit("🎉 视频拆帧处理完成！")
            self.parent_dialog.log_updated.emit("=" * 60)
            self.parent_dialog.log_updated.emit("📊 图片数量统计信息:")
            self.parent_dialog.log_updated.emit(f"   ├─ 理论生成图片数量: {image_counter} 张")
            self.parent_dialog.log_updated.emit(f"   ├─ 实际保存图片数量: {actual_count} 张")
            self.parent_dialog.log_updated.emit(f"   └─ 保存成功率: {(actual_count/image_counter*100):.1f}%" if image_counter > 0 else "   └─ 保存成功率: 0%")
            self.parent_dialog.log_updated.emit("")
            self.parent_dialog.log_updated.emit("📈 各视频文件图片统计:")
            
            # 显示每个视频的图片数量统计
            for video_file, count in results.items():
                video_name = os.path.basename(video_file)
                if count is not None and count > 0:
                    self.parent_dialog.log_updated.emit(f"   ├─ {video_name}: {count} 张图片")
                elif count == 0:
                    self.parent_dialog.log_updated.emit(f"   ├─ {video_name}: 0 张图片 (无符合条件的帧)")
                else:
                    self.parent_dialog.log_updated.emit(f"   ├─ {video_name}: 处理失败")
            
            self.parent_dialog.log_updated.emit("")
            self.parent_dialog.log_updated.emit("⏱️ 处理时间统计:")
            self.parent_dialog.log_updated.emit(f"   ├─ 总处理时间: {elapsed_formatted}")
            self.parent_dialog.log_updated.emit(f"   ├─ 处理视频数量: {len(processed_videos)}")
            self.parent_dialog.log_updated.emit(f"   └─ 平均每个视频: {elapsed_time/len(processed_videos):.1f} 秒" if len(processed_videos) > 0 else "   └─ 平均每个视频: 0 秒")
            
            # 计算处理效率
            if elapsed_time > 0:
                images_per_second = actual_count / elapsed_time
                self.parent_dialog.log_updated.emit("")
                self.parent_dialog.log_updated.emit("🚀 处理效率统计:")
                self.parent_dialog.log_updated.emit(f"   ├─ 图片生成速度: {images_per_second:.2f} 张/秒")
                self.parent_dialog.log_updated.emit(f"   └─ 帧处理速度: {self.total_frames_all_videos/elapsed_time:.0f} 帧/秒")
            
            self.parent_dialog.log_updated.emit("=" * 60)
            
            # 保存处理结果
            self.save_results(processed_videos, results, elapsed_formatted, actual_count)
            
            # 发送完成信号
            self.parent_dialog.extraction_finished.emit({
                'success': True,
                'stats': {
                    'elapsed_time': elapsed_formatted,
                    'video_count': len(processed_videos),
                    'image_count': actual_count
                }
            })
            
        except Exception as e:
            self.parent_dialog.extraction_finished.emit({
                'success': False,
                'error': str(e)
            })
    
    def calculate_total_frames(self):
        """计算所有视频的总帧数"""
        self.total_frames_all_videos = 0
        for video_file in self.video_files:
            if self.should_stop:
                break
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.total_frames_all_videos += total_frames
                cap.release()
        
        self.parent_dialog.log_updated.emit(f"总计需要处理 {self.total_frames_all_videos} 帧")
    
    def sanitize_filename(self, name):
        """清理文件名"""
        sanitized = re.sub(r'[^\w\-.\u4e00-\u9fff]', '_', name)
        return sanitized
    
    def process_video_with_progress(self, video_file_path, processed_videos, video_index, total_videos):
        """带进度显示的视频处理函数"""
        if self.should_stop:
            return 0
        
        video_file_name = os.path.basename(video_file_path)
        video_file_name_base = os.path.splitext(video_file_name)[0]
        sanitized_base = self.sanitize_filename(video_file_name_base)
        
        self.parent_dialog.log_updated.emit("-" * 50)
        self.parent_dialog.log_updated.emit(f"🎥 开始处理视频 ({video_index}/{total_videos}): {video_file_name}")
        
        processed_videos.append(video_file_name)
        
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            self.parent_dialog.log_updated.emit(f"❌ 无法打开视频文件: {video_file_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        saved_count = 0
        
        self.parent_dialog.log_updated.emit(f"📊 视频信息: 总帧数 {total_frames}, 预计生成图片 {total_frames // self.frame_interval + 1} 张")
        
        # 每处理100帧更新一次进度
        progress_update_interval = max(100, total_frames // 20)
        
        while True:
            if self.should_stop:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                img_filename = f"{sanitized_base}_frame{frame_count}.jpg"
                img_path = os.path.join(self.output_folder, img_filename)
                
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(frame_rgb)
                    image_pil.save(img_path)
                    
                    saved_count += 1
                    # 只在特定间隔显示保存信息，避免日志过多
                    if saved_count % 10 == 0 or saved_count == 1:
                        self.parent_dialog.log_updated.emit(f"💾 已保存第 {saved_count} 张图片: {img_filename}")
                    
                except Exception as e:
                    self.parent_dialog.log_updated.emit(f"❌ 保存图片失败: {e}")
            
            frame_count += 1
            self.processed_frames_all_videos += 1
            
            # 定期更新进度
            if frame_count % progress_update_interval == 0 or frame_count == total_frames:
                if self.total_frames_all_videos > 0:
                    overall_progress = int((self.processed_frames_all_videos / self.total_frames_all_videos) * 100)
                    self.parent_dialog.progress_updated.emit(min(overall_progress, 100))
                
                frame_progress = int((frame_count / total_frames) * 100)
                self.parent_dialog.log_updated.emit(f"⏳ 视频 {video_file_name} 处理进度: {frame_progress}% ({frame_count}/{total_frames} 帧)")
        
        cap.release()
        
        # 显示该视频的详细统计信息
        expected_images = total_frames // self.frame_interval + (1 if total_frames % self.frame_interval == 0 else 0)
        success_rate = (saved_count / expected_images * 100) if expected_images > 0 else 0
        
        self.parent_dialog.log_updated.emit(f"✅ 视频 {video_file_name} 处理完成")
        self.parent_dialog.log_updated.emit(f"   ├─ 实际生成图片: {saved_count} 张")
        self.parent_dialog.log_updated.emit(f"   ├─ 预期生成图片: {expected_images} 张")
        self.parent_dialog.log_updated.emit(f"   └─ 生成成功率: {success_rate:.1f}%")
        
        return saved_count
    
    def save_results(self, processed_videos, results, elapsed_time, actual_count):
        """保存处理结果到文件"""
        try:
            # 保存已处理的视频文件名
            output_txt_path = os.path.join(os.path.dirname(self.output_folder), "processed_videos.txt")
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                for video in processed_videos:
                    f.write(video + '\n')
            
            # 保存统计信息
            stats_path = os.path.join(os.path.dirname(self.output_folder), "processing_statistics.txt")
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("拆帧处理统计信息\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理总耗时: {elapsed_time}\n")
                f.write(f"处理视频数量: {len(processed_videos)}\n")
                f.write(f"输出图片数量: {actual_count}\n\n")
                
                f.write("各视频文件抽帧统计:\n")
                f.write("-" * 40 + "\n")
                for video, count in results.items():
                    video_name = os.path.basename(video)
                    if count is not None:
                        f.write(f"- {video_name}: {count} 张\n")
                    else:
                        f.write(f"- {video_name}: 处理失败\n")
                
                f.write("=" * 50 + "\n")
            
            self.parent_dialog.log_updated.emit(f"处理结果已保存到: {stats_path}")
            
        except Exception as e:
            self.parent_dialog.log_updated.emit(f"保存结果文件时出错: {e}")


def show_video_frame_extractor(parent=None):
    """显示视频拆帧工具对话框"""
    dialog = VideoFrameExtractorDialog(parent)
    return dialog.exec_()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    dialog = VideoFrameExtractorDialog()
    dialog.show()
    
    sys.exit(app.exec_())