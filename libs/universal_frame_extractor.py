import os
import cv2
import re
import time
import glob
from datetime import timedelta
from PIL import Image
import concurrent.futures
from multiprocessing import Manager
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QTextEdit, QProgressBar, QFileDialog, 
                             QMessageBox, QGroupBox, QSpinBox, QCheckBox, QFormLayout,
                             QScrollArea, QWidget, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont


class FrameExtractionWorker(QThread):
    """拆帧工作线程"""
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.should_stop = False
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        try:
            result = self.process_videos()
            self.finished.emit(result)
        except Exception as e:
            self.log_updated.emit(f"处理过程中发生错误: {str(e)}")
            self.finished.emit({"success": False, "error": str(e)})
    
    def sanitize_filename(self, name):
        """清理文件名，保留中文字符、字母、数字、下划线、短横线和点"""
        sanitized = re.sub(r'[^\w\-\.\u4e00-\u9fff]', '_', name)
        return sanitized
    
    def find_all_txt_files(self, root_path, recursive=True):
        """在指定目录下查找所有txt文件"""
        all_txt_files = []
        
        if recursive:
            for root, _, files in os.walk(root_path):
                for file in files:
                    if file.lower().endswith('.txt'):
                        all_txt_files.append(os.path.join(root, file))
        else:
            all_txt_files = glob.glob(os.path.join(root_path, "*.txt"))
        
        return all_txt_files
    
    def find_matching_video_files(self, txt_files, video_root_path):
        """对于每个txt文件，查找对应的视频文件"""
        matching_files = {}
        video_extensions = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        
        for txt_file in txt_files:
            txt_filename = os.path.basename(txt_file)
            base_name = os.path.splitext(txt_filename)[0]
            
            # 查找对应的视频文件
            for root, _, files in os.walk(video_root_path):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        file_base = os.path.splitext(file)[0]
                        if file_base == base_name:
                            video_path = os.path.join(root, file)
                            matching_files[txt_file] = video_path
                            break
                if txt_file in matching_files:
                    break
        
        return matching_files
    
    def process_video(self, video_file_path, txt_file_path, processed_videos, folder_paths, image_counters):
        """处理单个视频文件"""
        if self.should_stop:
            return
            
        video_file_name = os.path.basename(video_file_path)
        video_file_name_base = os.path.splitext(video_file_name)[0]
        sanitized_base = self.sanitize_filename(video_file_name_base)
        
        self.log_updated.emit(f"处理视频: {video_file_name}")
        
        if not os.path.exists(txt_file_path):
            self.log_updated.emit(f"对应的txt文件 {txt_file_path} 不存在，跳过此视频文件。")
            return
        
        processed_videos.append(video_file_path)
        clip_frames = []
        
        # 读取txt文件，提取帧范围和分类信息
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    self.log_updated.emit(f"txt文件 {txt_file_path} 中的行格式不正确: {line}")
                    continue
                
                try:
                    frame_range = parts[0]
                    category1_value = int(parts[1])
                    category2_value = int(parts[2]) if len(parts) > 2 else 0
                    start_frame, end_frame = map(int, frame_range.split('-'))
                except ValueError:
                    self.log_updated.emit(f"txt文件 {txt_file_path} 中的行格式不正确或数据类型错误: {line}")
                    continue
                
                clip_frames.append((start_frame, end_frame, category1_value, category2_value))
        
        if not clip_frames:
            self.log_updated.emit(f"视频 {video_file_name} 中没有有效的帧信息，跳过。")
            return
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            self.log_updated.emit(f"无法打开视频文件 {video_file_path}")
            return
        
        # 处理每个帧段
        for idx, (start_frame, end_frame, category1_value, category2_value) in enumerate(clip_frames):
            if self.should_stop:
                break
                
            total_frames = end_frame - start_frame + 1
            extraction_mode = self.config.get('extraction_mode', '平均帧数模式')
            
            # 根据拆帧模式确定要处理的帧
            if extraction_mode == '平均帧数模式':
                # 平均帧数模式 - 均匀分布取帧
                frames_per_segment = self.config.get('frames_per_segment', 15)
                if total_frames <= frames_per_segment:
                    # 如果总帧数小于等于需要的帧数，取所有帧
                    frames_to_process = list(range(start_frame, end_frame + 1))
                else:
                    # 均匀分布取帧
                    step = (total_frames - 1) / (frames_per_segment - 1)
                    frames_to_process = [start_frame + int(round(step * i)) for i in range(frames_per_segment)]
            else:
                # 固定帧数模式 - 按固定间隔取帧
                fixed_interval = self.config.get('fixed_frame_interval', 30)
                frames_to_process = []
                current_frame = start_frame
                while current_frame <= end_frame:
                    frames_to_process.append(current_frame)
                    current_frame += fixed_interval
            
            # 处理每一帧
            for i, frame_num in enumerate(frames_to_process):
                if self.should_stop:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    self.log_updated.emit(f"无法读取视频 {video_file_name} 的第 {frame_num} 帧")
                    continue
                
                # 确定保存路径 - 按照数字分类创建文件夹结构
                folder_path = os.path.join(self.config['output_path'], str(category1_value), str(category2_value))
                os.makedirs(folder_path, exist_ok=True)
                
                # 生成文件名
                img_filename = f"{sanitized_base}_frame{frame_num}_{category1_value}_{category2_value}.jpg"
                img_path = os.path.join(folder_path, img_filename)
                
                # 保存图片
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(frame_rgb)
                    image_pil.save(img_path)
                    
                    # 更新计数器
                    counter_key = f"{category1_value}_{category2_value}"
                    if counter_key not in image_counters:
                        image_counters[counter_key] = 0
                    image_counters[counter_key] += 1
                    
                except Exception as e:
                    self.log_updated.emit(f"保存图片 {img_path} 时出错: {e}")
                    continue
        
        cap.release()
    
    def process_videos(self):
        """处理所有视频"""
        config = self.config
        
        # 查找txt文件
        self.log_updated.emit("正在搜索txt文件...")
        all_txt_files = self.find_all_txt_files(config['txt_path'], config['recursive_search'])
        self.log_updated.emit(f"找到 {len(all_txt_files)} 个txt文件")
        
        # 查找匹配的视频文件
        self.log_updated.emit("正在查找对应的视频文件...")
        matching_files = self.find_matching_video_files(all_txt_files, config['video_path'])
        self.log_updated.emit(f"找到 {len(matching_files)} 对匹配的文件")
        
        if not matching_files:
            return {"success": False, "error": "没有找到匹配的txt和视频文件"}
        
        # 使用Manager创建共享对象
        with Manager() as manager:
            processed_videos = manager.list()
            image_counters = manager.dict()
            
            total_files = len(matching_files)
            processed_count = 0
            
            # 处理文件
            for txt_file, video_file in matching_files.items():
                if self.should_stop:
                    break
                    
                self.process_video(
                    video_file, txt_file,
                    processed_videos, None, image_counters
                )
                
                processed_count += 1
                progress = int((processed_count / total_files) * 100)
                self.progress_updated.emit(progress)
            
            # 统计结果
            total_images = sum(image_counters.values())
            
            return {
                "success": True,
                "processed_videos": len(processed_videos),
                "total_images": total_images,
                "image_counters": dict(image_counters)
            }


class UniversalFrameExtractionDialog(QDialog):
    """通用视频拆帧对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('通用视频拆帧工具')
        self.setModal(True)
        self.resize(900, 850)  # 增大界面尺寸，避免滚动
        
        self.worker = None
        self.initUI()
        self.load_default_config()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 路径配置组
        path_group = QGroupBox("路径配置")
        path_layout = QFormLayout(path_group)
        
        # txt文件路径
        self.txt_path_edit = QLineEdit()
        txt_path_layout = QHBoxLayout()
        txt_path_layout.addWidget(self.txt_path_edit)
        txt_browse_btn = QPushButton("浏览")
        txt_browse_btn.clicked.connect(lambda: self.browse_folder(self.txt_path_edit))
        txt_path_layout.addWidget(txt_browse_btn)
        path_layout.addRow("TXT文件路径:", txt_path_layout)
        
        # 视频文件路径
        self.video_path_edit = QLineEdit()
        video_path_layout = QHBoxLayout()
        video_path_layout.addWidget(self.video_path_edit)
        video_browse_btn = QPushButton("浏览")
        video_browse_btn.clicked.connect(lambda: self.browse_folder(self.video_path_edit))
        video_path_layout.addWidget(video_browse_btn)
        path_layout.addRow("视频文件路径:", video_path_layout)
        
        # 输出路径
        self.output_path_edit = QLineEdit()
        output_path_layout = QHBoxLayout()
        output_path_layout.addWidget(self.output_path_edit)
        output_browse_btn = QPushButton("浏览")
        output_browse_btn.clicked.connect(lambda: self.browse_folder(self.output_path_edit))
        output_path_layout.addWidget(output_browse_btn)
        path_layout.addRow("输出路径:", output_path_layout)
        
        # 递归搜索选项
        self.recursive_checkbox = QCheckBox("递归搜索子文件夹")
        self.recursive_checkbox.setChecked(True)
        path_layout.addRow("", self.recursive_checkbox)
        
        layout.addWidget(path_group)
        
        # 拆帧配置组
        frame_group = QGroupBox("拆帧配置")
        frame_layout = QFormLayout(frame_group)
        
        # 拆帧模式选择
        self.extraction_mode_combo = QComboBox()
        self.extraction_mode_combo.addItems(["平均帧数模式", "固定帧数模式"])
        self.extraction_mode_combo.currentTextChanged.connect(self.on_mode_changed)
        frame_layout.addRow("拆帧模式:", self.extraction_mode_combo)
        
        # 平均帧数配置
        self.frames_per_segment_spin = QSpinBox()
        self.frames_per_segment_spin.setRange(1, 100)
        self.frames_per_segment_spin.setValue(15)
        frame_layout.addRow("平均帧数 (每个区间均匀取帧数量):", self.frames_per_segment_spin)
        
        # 固定帧数配置
        self.fixed_frame_interval_spin = QSpinBox()
        self.fixed_frame_interval_spin.setRange(1, 1000)
        self.fixed_frame_interval_spin.setValue(30)
        self.fixed_frame_interval_spin.setEnabled(False)  # 默认禁用
        frame_layout.addRow("固定帧数 (区间每隔多少帧取一帧):", self.fixed_frame_interval_spin)
        
        # 移除说明文本，界面更简洁
        
        layout.addWidget(frame_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 日志区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始拆帧")
        self.start_button.clicked.connect(self.start_extraction)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        button_layout.addStretch()
        
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def browse_folder(self, line_edit):
        """浏览文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            line_edit.setText(folder)
    
    def on_mode_changed(self, mode_text):
        """拆帧模式切换回调函数"""
        if mode_text == "平均帧数模式":
            # 启用平均帧数，禁用固定帧数
            self.frames_per_segment_spin.setEnabled(True)
            self.fixed_frame_interval_spin.setEnabled(False)
        elif mode_text == "固定帧数模式":
            # 启用固定帧数，禁用平均帧数
            self.frames_per_segment_spin.setEnabled(False)
            self.fixed_frame_interval_spin.setEnabled(True)
    
    def load_default_config(self):
        """加载默认配置"""
        # 不需要加载默认配置，用户直接设置平均帧数即可
        pass
    
    def validate_config(self):
        """验证配置"""
        if not self.txt_path_edit.text().strip():
            QMessageBox.warning(self, "配置错误", "请选择TXT文件路径")
            return False
        
        if not self.video_path_edit.text().strip():
            QMessageBox.warning(self, "配置错误", "请选择视频文件路径")
            return False
        
        if not self.output_path_edit.text().strip():
            QMessageBox.warning(self, "配置错误", "请选择输出路径")
            return False
        
        # 根据模式验证相应的参数
        mode = self.extraction_mode_combo.currentText()
        if mode == "平均帧数模式":
            if self.frames_per_segment_spin.value() <= 0:
                QMessageBox.warning(self, "配置错误", "平均帧数必须大于0")
                return False
        elif mode == "固定帧数模式":
            if self.fixed_frame_interval_spin.value() <= 0:
                QMessageBox.warning(self, "配置错误", "固定帧数间隔必须大于0")
                return False
        
        return True
    
    def get_config(self):
        """获取配置"""
        config = {
            'txt_path': self.txt_path_edit.text().strip(),
            'video_path': self.video_path_edit.text().strip(),
            'output_path': self.output_path_edit.text().strip(),
            'recursive_search': self.recursive_checkbox.isChecked(),
            'extraction_mode': self.extraction_mode_combo.currentText(),
            'frames_per_segment': self.frames_per_segment_spin.value(),
            'fixed_frame_interval': self.fixed_frame_interval_spin.value()
        }
        return config
    
    def start_extraction(self):
        """开始拆帧"""
        if not self.validate_config():
            return
        
        config = self.get_config()
        
        # 清空日志
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        self.worker = FrameExtractionWorker(config)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_updated.connect(self.append_log)
        self.worker.finished.connect(self.extraction_finished)
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 开始处理
        self.worker.start()
        self.append_log("开始拆帧处理...")
    
    def stop_extraction(self):
        """停止拆帧"""
        if self.worker:
            self.worker.stop()
            self.append_log("正在停止处理...")
    
    def append_log(self, message):
        """添加日志"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def extraction_finished(self, result):
        """拆帧完成"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if result.get("success", False):
            self.append_log(f"拆帧完成！")
            self.append_log(f"处理视频数量: {result.get('processed_videos', 0)}")
            self.append_log(f"生成图片总数: {result.get('total_images', 0)}")
            
            # 显示详细统计
            image_counters = result.get('image_counters', {})
            if image_counters:
                self.append_log("详细统计:")
                for key, count in image_counters.items():
                    cat1_value, cat2_value = key.split('_')
                    self.append_log(f"  分类 {cat1_value}-{cat2_value}: {count} 张")
            
            QMessageBox.information(self, "完成", "拆帧处理完成！")
        else:
            error_msg = result.get("error", "未知错误")
            self.append_log(f"拆帧失败: {error_msg}")
            QMessageBox.critical(self, "错误", f"拆帧失败: {error_msg}")
        
        self.worker = None


# 便捷函数，供外部调用
def open_frame_extraction_dialog(parent=None):
    """打开拆帧对话框"""
    dialog = UniversalFrameExtractionDialog(parent)
    return dialog.exec_()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    dialog = UniversalFrameExtractionDialog()
    dialog.show()
    sys.exit(app.exec_())