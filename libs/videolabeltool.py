import sys
import os
import cv2
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QSlider,
                             QListWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QStyle,
                             QMessageBox, QShortcut, QDialog, QComboBox, QDialogButtonBox,
                             QProgressBar, QLineEdit, QSizePolicy, QAbstractItemView,
                             QGroupBox, QTextEdit, QListWidgetItem)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QEvent, QThread, pyqtSignal, QRect
from PyQt5.QtWidgets import QMenu
import copy
try:
    # 尝试相对导入（从包内调用时）
    from .universal_frame_extractor import UniversalFrameExtractionDialog
except ImportError:
    # 如果相对导入失败，尝试绝对导入（直接运行时）
    from universal_frame_extractor import UniversalFrameExtractionDialog

class LabelListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def focusOutEvent(self, event):
        self.clearSelection()
        super().focusOutEvent(event)

class LabelSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.labels = []
        self.selected_label_index = None

    def set_labels(self, labels):
        self.labels = labels
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        for index, label in enumerate(self.labels):
            pos = (label['frame'] / self.maximum()) * self.width()
            painter.drawLine(int(pos), 0, int(pos), self.height())
        if self.selected_label_index is not None and 0 <= self.selected_label_index < len(self.labels):
            selected_label = self.labels[self.selected_label_index]
            pos = (selected_label['frame'] / self.maximum()) * self.width()
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawLine(int(pos), 0, int(pos), self.height())

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

# 标签配置对话框 - 用于设置自定义分类
class LabelConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('标签配置')
        self.setModal(True)
        self.resize(500, 400)
        self.initUI()
        
        # 默认配置（空配置，完全由用户自定义）
        self.category1_name = ""
        self.category1_options = []
        self.category2_name = ""
        self.category2_options = []
        
        self.load_default_config()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 说明文字 - 移除小字体设置，使用默认字体大小
        info_label = QLabel("配置标签分类，第一个分类必填，第二个分类可选（留空表示不使用）")
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # 第一个分类配置
        category1_group = QGroupBox("第一个分类（必填）")
        category1_layout = QVBoxLayout(category1_group)
        
        # 分类名称标签
        name_label1 = QLabel("分类名称:")
        category1_layout.addWidget(name_label1)
        
        # 分类名称输入框
        self.category1_name_edit = QLineEdit()
        self.category1_name_edit.setPlaceholderText("例如: 状态分类")
        category1_layout.addWidget(self.category1_name_edit)
        
        # 选项列表标签
        options_label1 = QLabel("选项列表 (用逗号分隔):")
        category1_layout.addWidget(options_label1)
        
        # 选项列表输入框
        self.category1_options_edit = QTextEdit()
        self.category1_options_edit.setMaximumHeight(80)
        self.category1_options_edit.setPlaceholderText("例如: 选项1,选项2,选项3")
        category1_layout.addWidget(self.category1_options_edit)
        
        layout.addWidget(category1_group)
        
        # 第二个分类配置
        category2_group = QGroupBox("第二个分类（可选）")
        category2_layout = QVBoxLayout(category2_group)
        
        # 分类名称标签
        name_label2 = QLabel("分类名称:")
        category2_layout.addWidget(name_label2)
        
        # 分类名称输入框
        self.category2_name_edit = QLineEdit()
        self.category2_name_edit.setPlaceholderText("例如: 类型分类（留空表示不使用第二个分类）")
        category2_layout.addWidget(self.category2_name_edit)
        
        # 选项列表标签
        options_label2 = QLabel("选项列表 (用逗号分隔):")
        category2_layout.addWidget(options_label2)
        
        # 选项列表输入框
        self.category2_options_edit = QTextEdit()
        self.category2_options_edit.setMaximumHeight(80)
        self.category2_options_edit.setPlaceholderText("例如: 类型A,类型B,类型C,类型D")
        category2_layout.addWidget(self.category2_options_edit)
        
        layout.addWidget(category2_group)
        
        # 按钮区域
        buttons_layout = QHBoxLayout()
        
        # 恢复默认按钮
        self.reset_button = QPushButton("恢复默认")
        self.reset_button.clicked.connect(self.load_default_config)
        buttons_layout.addWidget(self.reset_button)
        
        # 添加弹性空间
        buttons_layout.addStretch()
        
        # 确定按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.ok_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_button)
        
        layout.addLayout(buttons_layout)

    def load_default_config(self):
        """加载默认配置"""
        self.category1_name_edit.setText(self.category1_name)
        self.category1_options_edit.setPlainText(",".join(self.category1_options))
        self.category2_name_edit.setText(self.category2_name)
        self.category2_options_edit.setPlainText(",".join(self.category2_options))

    def load_config(self, config):
        """加载指定的配置"""
        self.category1_name_edit.setText(config.get('category1_name', ''))
        self.category1_options_edit.setPlainText(",".join(config.get('category1_options', [])))
        self.category2_name_edit.setText(config.get('category2_name', ''))
        self.category2_options_edit.setPlainText(",".join(config.get('category2_options', [])))

    def get_config(self):
        """获取配置结果"""
        category1_name = self.category1_name_edit.text().strip()
        category1_options = [opt.strip() for opt in self.category1_options_edit.toPlainText().split(",") if opt.strip()]
        category2_name = self.category2_name_edit.text().strip()
        category2_options = [opt.strip() for opt in self.category2_options_edit.toPlainText().split(",") if opt.strip()]
        
        return {
            'category1_name': category1_name,
            'category1_options': category1_options,
            'category2_name': category2_name,
            'category2_options': category2_options
        }

    def accept(self):
        """验证配置并接受"""
        config = self.get_config()
        
        # 验证第一个分类（必填）
        if not config['category1_name']:
            QMessageBox.warning(self, '配置错误', '请输入第一个分类的名称')
            return
        if not config['category1_options']:
            QMessageBox.warning(self, '配置错误', '请输入第一个分类的选项')
            return
            
        # 验证第二个分类（可选，但如果填写了名称就必须填写选项）
        if config['category2_name'] and not config['category2_options']:
            QMessageBox.warning(self, '配置错误', '第二个分类已填写名称，请输入对应的选项，或者清空名称表示不使用第二个分类')
            return
        if not config['category2_name'] and config['category2_options']:
            QMessageBox.warning(self, '配置错误', '第二个分类已填写选项，请输入对应的名称，或者清空选项表示不使用第二个分类')
            return
            
        super().accept()


# 通用标签对话框 - 支持自定义分类
class LabelDialog(QDialog):
    def __init__(self, parent=None, label_config=None):
        super().__init__(parent)
        self.setWindowTitle('打标签')
        
        # 如果没有提供配置，使用空的默认配置，完全由用户自定义
        if label_config is None:
            self.label_config = {
                'category1_name': '',
                'category1_options': [],
                'category2_name': '',
                'category2_options': []
            }
        else:
            self.label_config = label_config
            
        self.initUI()
        self.category1_value = None
        self.category2_value = None
        self.is_condition_end = False

    def initUI(self):
        layout = QVBoxLayout(self)

        # 第一个分类（始终显示）
        self.category1_label = QLabel(f'{self.label_config["category1_name"]}:')
        layout.addWidget(self.category1_label)
        
        self.category1_combo = QComboBox()
        self.category1_combo.addItems(self.label_config['category1_options'])
        layout.addWidget(self.category1_combo)

        # 第二个分类（根据配置决定是否显示）
        self.has_category2 = bool(self.label_config.get('category2_name') and self.label_config.get('category2_options'))
        
        if self.has_category2:
            self.category2_label = QLabel(f'{self.label_config["category2_name"]}:')
            layout.addWidget(self.category2_label)
            
            self.category2_combo = QComboBox()
            self.category2_combo.addItems(self.label_config['category2_options'])
            layout.addWidget(self.category2_combo)
        else:
            # 如果没有第二个分类，创建空的占位符
            self.category2_label = None
            self.category2_combo = None

        # 按钮
        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton('确定', QDialogButtonBox.AcceptRole)
        self.cancel_button = buttons.addButton('取消', QDialogButtonBox.RejectRole)
        self.condition_end_button = buttons.addButton('结束标记', QDialogButtonBox.ActionRole)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.condition_end_button.clicked.connect(self.condition_end)
        layout.addWidget(buttons)

        # 设置快捷键
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """设置快捷键"""
        # Shift+Enter 表示工况结束
        self.shortcut_shift_enter = QShortcut(QKeySequence(Qt.SHIFT + Qt.Key_Return), self)
        self.shortcut_shift_enter.activated.connect(self.condition_end)

        # 数字键1-9选择第一个分类的选项
        for i in range(min(9, len(self.label_config['category1_options']))):
            shortcut = QShortcut(QKeySequence(str(i + 1)), self)
            shortcut.activated.connect(lambda index=i: self.category1_combo.setCurrentIndex(index))

        # 字母键q,w,e,r,t,y,u,i,o,p选择第二个分类的选项（仅在有第二个分类时设置）
        if self.has_category2:
            keys = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
            for i, key in enumerate(keys):
                if i < len(self.label_config['category2_options']):
                    shortcut = QShortcut(QKeySequence(key), self)
                    shortcut.activated.connect(lambda index=i: self.category2_combo.setCurrentIndex(index))

    def accept(self):
        """确定按钮点击事件"""
        self.category1_value = self.category1_combo.currentText()
        
        # 只有在有第二个分类时才获取第二个分类的值
        if self.has_category2:
            self.category2_value = self.category2_combo.currentText()
        else:
            self.category2_value = None  # 没有第二个分类时设为None
            
        self.is_condition_end = False
        super().accept()

    def condition_end(self):
        """结束标记按钮点击事件"""
        self.is_condition_end = True
        super().accept()

class VideoLabelingTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('视频打标签工具')
        self.showMaximized()  # 设置窗口最大化显示
        self.initUI()
        self.initVariables()
        self.initShortcuts()
        self.setFocusPolicy(Qt.StrongFocus)

    def initUI(self):
        main_layout = QHBoxLayout(self)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label, stretch=5)

        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        self.load_button = QPushButton('加载视频')
        self.load_button.clicked.connect(self.load_video)
        self.load_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.load_button)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_pause_video)
        self.play_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.play_button)

        self.speed_buttons = []
        # 定义倍速值和对应的显示文字
        speed_configs = [
            (0.5, '0.5x'),  # 0.5倍速显示为"0.5x"
            (1, '1x'),      # 1倍速显示为"1x"
            (2, '2x'),      # 2倍速显示为"2x"
            (5, '5x')       # 5倍速显示为"5x"
        ]
        
        for speed, display_text in speed_configs:
            btn = QPushButton(display_text)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked, s=speed: self.set_playback_speed(s))
            controls_layout.addWidget(btn)
            self.speed_buttons.append(btn)

        # 独立的拆帧按钮
        self.frame_extract_button = QPushButton('拆帧')
        self.frame_extract_button.clicked.connect(self.open_frame_extraction_dialog)
        self.frame_extract_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.frame_extract_button)

        self.label_button = QPushButton('打标签')
        self.label_button.clicked.connect(self.add_label)
        self.label_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.label_button)

        self.export_button = QPushButton('导出TXT')
        self.export_button.clicked.connect(self.export_labels)
        self.export_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.export_button)

        self.import_button = QPushButton('导入TXT')
        self.import_button.clicked.connect(self.import_labels)
        self.import_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.import_button)

        self.config_button = QPushButton('标签配置')
        self.config_button.clicked.connect(self.configure_labels)
        self.config_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls_layout.addWidget(self.config_button)

        left_layout.addWidget(controls_widget, stretch=0)

        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)

        self.progress_slider = LabelSlider(Qt.Horizontal)
        self.progress_slider.sliderMoved.connect(self.set_frame_position)
        self.progress_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        progress_layout.addWidget(self.progress_slider)

        self.time_label = QLabel('00:00:00/00:00:00')
        self.time_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        progress_layout.addWidget(self.time_label)

        self.frame_label = QLabel('帧: 0')
        self.frame_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        progress_layout.addWidget(self.frame_label)

        left_layout.addWidget(progress_widget, stretch=0)

        main_layout.addWidget(left_widget, stretch=5)

        self.label_list = LabelListWidget()
        self.label_list.setMaximumWidth(300)
        self.label_list.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.label_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.label_list.itemClicked.connect(self.label_clicked)
        main_layout.addWidget(self.label_list, stretch=1)

        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(self.show_label_context_menu)

        self.video_label.setFocusPolicy(Qt.ClickFocus)
        self.setLayout(main_layout)

    def initVariables(self):
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.playback_speed = 1.0
        self.frame_count = 0
        self.fps = 0
        self.duration = 0
        self.current_frame = 0
        self.labels = []
        self.video_loaded = False
        self.last_frame_time = 0
        self.skip_frames = 1
        self.video_file = None
        self.video_filename = None
        
        # 配置文件路径
        self.config_file = os.path.join(os.path.dirname(__file__), 'label_config.json')
        
        # 标签配置 - 从配置文件加载或使用默认配置
        self.label_config = self.load_label_config()

    def initShortcuts(self):
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.handle_left_right_keys(-1))
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.handle_left_right_keys(1))

        self.shortcut_ctrl_left = QShortcut(QKeySequence(Qt.ControlModifier + Qt.Key_Left), self)
        self.shortcut_ctrl_left.activated.connect(lambda: self.handle_left_right_keys(-50))
        self.shortcut_ctrl_right = QShortcut(QKeySequence(Qt.ControlModifier + Qt.Key_Right), self)
        self.shortcut_ctrl_right.activated.connect(lambda: self.handle_left_right_keys(50))

        self.shortcut_ctrl_alt_left = QShortcut(QKeySequence(Qt.ControlModifier + Qt.AltModifier + Qt.Key_Left), self)
        self.shortcut_ctrl_alt_left.activated.connect(lambda: self.handle_left_right_keys(-200))
        self.shortcut_ctrl_alt_right = QShortcut(QKeySequence(Qt.ControlModifier + Qt.AltModifier + Qt.Key_Right), self)
        self.shortcut_ctrl_alt_right.activated.connect(lambda: self.handle_left_right_keys(200))

        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.activated.connect(self.play_pause_video)

        self.shortcut_enter = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.shortcut_enter.activated.connect(self.add_label)

        self.shortcut_delete = QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.shortcut_delete.activated.connect(self.delete_selected_label)

    def load_label_config(self):
        """从配置文件加载标签配置"""
        default_config = {
            'category1_name': '',
            'category1_options': [],
            'category2_name': '',
            'category2_options': []
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 验证配置格式
                    if all(key in config for key in default_config.keys()):
                        return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"加载配置文件失败: {e}")
        
        return default_config

    def save_label_config(self):
        """保存标签配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.label_config, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"保存配置文件失败: {e}")
            QMessageBox.warning(self, '保存失败', f'无法保存配置文件：{e}')

    def load_video(self):
        # 支持多种主流视频格式的文件过滤器
        video_filter = (
            "视频文件 (*.avi *.mp4 *.mov *.mkv *.wmv *.flv *.webm);;"
            "AVI 文件 (*.avi);;"
            "MP4 文件 (*.mp4);;"
            "MOV 文件 (*.mov);;"
            "MKV 文件 (*.mkv);;"
            "WMV 文件 (*.wmv);;"
            "FLV 文件 (*.flv);;"
            "WebM 文件 (*.webm);;"
            "所有文件 (*.*)"
        )
        video_file, _ = QFileDialog.getOpenFileName(self, '打开视频文件', '', video_filter)
        if video_file:
            # 获取文件信息
            self.video_file = video_file
            self.video_filename = os.path.splitext(os.path.basename(video_file))[0]
            file_extension = os.path.splitext(video_file)[1].lower()
            
            # 检查文件是否存在
            if not os.path.exists(video_file):
                QMessageBox.critical(self, '文件错误', f'视频文件不存在：\n{video_file}')
                return
            
            # 尝试打开视频文件
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                error_msg = (
                    f'无法打开视频文件：\n{os.path.basename(video_file)}\n\n'
                    f'文件格式：{file_extension}\n\n'
                    '可能的原因：\n'
                    '• 视频文件已损坏\n'
                    '• 缺少相应的解码器\n'
                    '• 文件格式不受支持\n\n'
                    '支持的格式：AVI, MP4, MOV, MKV, WMV, FLV, WebM'
                )
                QMessageBox.critical(self, '视频加载失败', error_msg)
                return
            
            # 获取视频属性
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # 验证视频属性
            if self.fps == 0 or self.frame_count == 0:
                error_msg = (
                    f'视频文件信息异常：\n{os.path.basename(video_file)}\n\n'
                    f'帧率：{self.fps} FPS\n'
                    f'总帧数：{self.frame_count}\n\n'
                    '请检查视频文件是否完整或尝试其他格式的视频文件。'
                )
                QMessageBox.critical(self, '视频属性错误', error_msg)
                self.cap.release()
                return
            # 计算视频时长和初始化界面
            self.duration = self.frame_count / self.fps
            self.progress_slider.setMaximum(self.frame_count - 1)
            self.current_frame = 0
            self.playing = False
            self.video_loaded = True
            self.update_time_label()
            self.display_frame()
            self.skip_frames = 1
            self.labels = []
            self.update_label_list()
            self.progress_slider.set_labels(self.labels)
            
            # 移除修改模式相关代码
            
            # 显示视频加载成功信息
            video_info = (
                f'视频加载成功！\n\n'
                f'文件名：{os.path.basename(video_file)}\n'
                f'格式：{file_extension.upper()}\n'
                f'分辨率：{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))} × {int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}\n'
                f'帧率：{self.fps:.2f} FPS\n'
                f'总帧数：{self.frame_count:,}\n'
                f'时长：{self.format_time(self.duration)}\n\n'
                '现在可以开始标注工作了！'
            )
            QMessageBox.information(self, '加载成功', video_info)

    def display_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        else:
            self.timer.stop()
            self.playing = False
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(self.current_frame)
        self.progress_slider.blockSignals(False)
        self.update_time_label()
        self.update_progress_slider_labels()

    def update_time_label(self):
        current_time = self.current_frame / self.fps if self.fps != 0 else 0
        total_time = self.duration
        self.time_label.setText(f'{self.format_time(current_time)}/{self.format_time(total_time)}')
        self.frame_label.setText(f'帧: {self.current_frame}')

    def format_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '%02d:%02d:%02d' % (h, m, s)

    def play_pause_video(self):
        if not self.video_loaded:
            return
        if self.playing:
            self.timer.stop()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playing = False
        else:
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.playing = True

    def next_frame(self):
        if self.current_frame >= self.frame_count - 1:
            self.timer.stop()
            self.playing = False
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            return
        self.current_frame += int(self.playback_speed)
        self.current_frame = min(self.current_frame, self.frame_count - 1)
        self.display_frame()

    def set_playback_speed(self, speed):
        self.playback_speed = speed
        if self.playing:
            self.timer.stop()
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))

    def set_frame_position(self, position):
        if not self.video_loaded:
            return
        self.current_frame = position
        self.display_frame()

    def seek_frames(self, frames):
        if not self.video_loaded:
            return
        self.current_frame = max(0, min(self.current_frame + frames, self.frame_count - 1))
        self.display_frame()

    def configure_labels(self):
        """打开标签配置对话框"""
        dialog = LabelConfigDialog(self)
        # 加载当前配置到对话框
        dialog.load_config(self.label_config)
        
        result = dialog.exec_()
        if result == QDialog.Accepted:
            # 获取新的配置
            new_config = dialog.get_config()
            
            # 如果有现有标签，询问用户是否要清空
            if self.labels:
                reply = QMessageBox.question(
                    self, '配置更改', 
                    '更改标签配置将清空现有的所有标签，是否继续？',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.labels.clear()
                    self.update_label_list()
                    self.progress_slider.set_labels(self.labels)
                else:
                    return
            
            # 更新配置
            self.label_config = new_config
            # 保存配置到文件
            self.save_label_config()
            QMessageBox.information(self, '配置成功', '标签配置已更新并保存！')

    def add_label(self):
        """添加标签"""
        if not self.video_loaded:
            return
            
        self.shortcut_enter.setEnabled(False)
        dialog = LabelDialog(self, self.label_config)
        result = dialog.exec_()
        self.shortcut_enter.setEnabled(True)
        
        frame_number = self.current_frame
        
        if result == QDialog.Accepted:
            if dialog.is_condition_end:
                # 结束标记
                label = {
                    'frame': frame_number,
                    'category1_value': '结束标记',
                    'category2_value': None,
                    'is_condition_end': True
                }
            else:
                # 普通标签
                category1_value = dialog.category1_value
                category2_value = dialog.category2_value  # 可能为None
                
                label = {
                    'frame': frame_number,
                    'category1_value': category1_value,
                    'category2_value': category2_value,
                    'is_condition_end': False
                }
                
            # 直接添加标签，不进行逻辑判断
            if self.can_add_label(label):
                self.labels.append(label)
                self.labels.sort(key=lambda x: x['frame'])
                self.update_label_list()
                self.progress_slider.set_labels(self.labels)
            else:
                QMessageBox.warning(self, '无效的标签', '由于逻辑限制，无法添加此标签。')

    def can_add_label(self, new_label):
        temp_labels = copy.deepcopy(self.labels)
        temp_labels.append(new_label)
        temp_labels.sort(key=lambda x: x['frame'])
        return self.is_label_list_valid(temp_labels)

    def is_label_list_valid(self, labels):
        """验证标签列表的有效性 - 简化为基本检查"""
        labels_sorted = sorted(labels, key=lambda x: x['frame'])
        
        # 检查标签是否重合或在同一帧
        for i in range(1, len(labels_sorted)):
            if labels_sorted[i]['frame'] - labels_sorted[i - 1]['frame'] <= 0:
                return False
        
        # 对于通用标签系统，只进行基本的帧位置检查
        # 不再区分默认溢锅配置，统一使用简化的验证逻辑
        return True
    
    def _validate_boiling_logic(self, labels_sorted):
        """验证标签逻辑（通用版本）"""
        # 在通用标签系统中，不进行特定的逻辑验证
        # 让用户完全自定义标签逻辑，只进行基本的数据完整性检查
        
        # 检查标签数据的基本完整性
        for label in labels_sorted:
            # 检查必要的字段是否存在
            if 'frame' not in label:
                return False
            
            # 兼容新旧标签格式，检查是否有分类值
            category1_value = label.get('category1_value') or label.get('boiling_state')
            if not category1_value:
                return False
        
        return True

    def is_progressive(self, states):
        """检查状态是否递进 - 简化为通用逻辑"""
        # 不再进行特定的递进性检查，统一返回True
        # 让用户完全自定义标签逻辑
        return True

    def update_label_list(self):
        """更新标签列表显示"""
        self.label_list.clear()
        
        # 获取当前标签配置
        has_category2 = bool(self.label_config.get('category2_name') and self.label_config.get('category2_options'))
        
        for label in self.labels:
            # 计算当前帧对应的时间
            frame_time = self.format_time(label['frame'] / self.fps) if self.fps != 0 else '00:00:00'
            
            # 检查是否为结束标记（优先检查，避免被其他逻辑覆盖）
            is_end_marker = (label.get('is_condition_end', False) or 
                           label.get('category1_value') == '工况结束' or 
                           label.get('category1_value') == '结束标记' or
                           label.get('boiling_state') == '工况结束')
            
            if is_end_marker:
                display_text = f"帧: {label['frame']} ({frame_time}) - 结束标记"
            else:
                # 优先使用新的通用标签格式
                if 'category1_value' in label:
                    category1_text = label.get('category1_value', '')
                    
                    # 根据是否有第二个分类决定显示格式
                    if has_category2:
                        category2_text = label.get('category2_value', '')
                        if category2_text:  # 如果第二个分类有值，显示两个分类
                            display_text = f"帧: {label['frame']} ({frame_time}) - {category1_text}, {category2_text}"
                        else:  # 如果第二个分类为空，只显示第一个分类
                            display_text = f"帧: {label['frame']} ({frame_time}) - {category1_text}"
                    else:
                        display_text = f"帧: {label['frame']} ({frame_time}) - {category1_text}"
                # 兼容旧的溢锅标签格式
                else:
                    boiling_state = label.get('boiling_state', '')
                    lid_type = label.get('lid_type', '')
                    display_text = f"帧: {label['frame']} ({frame_time}) - {boiling_state}, {lid_type}"
                
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, label)
            self.label_list.addItem(item)
        self.progress_slider.set_labels(self.labels)

    def label_clicked(self, item):
        index = self.label_list.row(item)
        label = self.labels[index]
        self.current_frame = label['frame']
        self.display_frame()

    def delete_selected_label(self):
        selected_items = self.label_list.selectedItems()
        if selected_items:
            index = self.label_list.row(selected_items[0])
            temp_labels = copy.deepcopy(self.labels)
            del temp_labels[index]
            # 直接删除标签，不进行逻辑判断
            if self.is_label_list_valid(temp_labels):
                del self.labels[index]
                self.update_label_list()
                self.progress_slider.set_labels(self.labels)
            else:
                QMessageBox.warning(self, '删除无效', '删除该标签将导致标签逻辑错误，无法删除。')

    def show_label_context_menu(self, position):
        item = self.label_list.itemAt(position)
        if item:
            self.label_list.setCurrentItem(item)
            menu = QMenu(self)
            delete_action = menu.addAction("删除标签")
            action = menu.exec_(self.label_list.mapToGlobal(position))
            if action == delete_action:
                self.delete_selected_label()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_selected_label()
        elif event.key() == Qt.Key_Escape:
            self.label_list.clearSelection()
        else:
            super().keyPressEvent(event)

    def export_labels(self):
        """导出标签到TXT文件"""
        if not self.labels:
            QMessageBox.warning(self, '警告', '没有标签可导出。')
            return

        # 移除修改模式相关限制

        # 统一使用通用标签导出逻辑，不再区分默认溢锅配置
        self._export_generic_labels()

    def _export_boiling_labels(self):
        """导出溢锅标签 - 生成完整的帧区间用于视频剪辑，使用数字编码"""
        lines = []
        
        # 处理标签数据，生成连续的帧区间
        temp_labels = self.labels.copy()
        temp_labels.sort(key=lambda x: x['frame'])
        
        if not temp_labels:
            self._save_export_file(lines)
            return
        
        # 使用当前标签配置的选项（用于数字编码）
        category1_options = self.label_config.get('category1_options', [])
        category2_options = self.label_config.get('category2_options', [])
        
        # 生成连续的片段区间
        segments = []
        
        for i, label in enumerate(temp_labels):
            label_frame = label['frame']
            # 兼容新旧标签格式
            boiling_state = label.get('category1_value') or label.get('boiling_state')
            lid_type = label.get('category2_value') or label.get('lid_type')
            
            # 检查是否为结束标记（兼容旧版"工况结束"和新版结束标记）
            is_end_marker = (label.get('is_condition_end', False) or 
                           boiling_state == '工况结束')
            
            # 如果当前标签不是结束标记，创建一个片段
            if not is_end_marker:
                # 确定片段的结束帧
                if i + 1 < len(temp_labels):
                    # 下一个标签的前一帧作为结束帧
                    segment_end = temp_labels[i + 1]['frame'] - 1
                else:
                    # 最后一个标签，延续到视频结束
                    segment_end = self.frame_count - 1
                
                # 将分类转换为数字编码
                category1_code = self._category_to_code(boiling_state, category1_options)
                category2_code = self._category_to_code(lid_type, category2_options) if lid_type else 0
                
                # 创建片段记录
                segment = {
                    'start_frame': label_frame,
                    'end_frame': segment_end,
                    'category1_code': category1_code,
                    'category2_code': category2_code
                }
                segments.append(segment)
        
        # 输出片段信息 - 只输出帧区间和数字编码，不包含时间
        for segment in segments:
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            category1_code = segment['category1_code']
            category2_code = segment['category2_code']
            
            # 输出格式：起始帧-结束帧 第一分类编码 第二分类编码
            lines.append(f"{start_frame}-{end_frame} {category1_code} {category2_code}")
        
        self._save_export_file(lines)

    def _export_generic_labels(self):
        """导出通用标签 - 简化格式：帧区间在第一行，使用数字编码"""
        lines = []
        
        # 处理标签数据
        temp_labels = self.labels.copy()
        temp_labels.sort(key=lambda x: x['frame'])
        
        if not temp_labels:
            self._save_export_file(lines)
            return
        
        # 获取分类选项用于数字编码
        category1_options = self.label_config.get('category1_options', [])
        category2_options = self.label_config.get('category2_options', [])
        
        # 生成连续的片段区间（非结束标记的标签）
        segments = []
        
        for i, label in enumerate(temp_labels):
            label_frame = label['frame']
            category1 = label.get('category1_value', '')
            category2 = label.get('category2_value', '')
            
            # 检查是否为结束标记（兼容旧版"工况结束"和新版结束标记）
            is_end_marker = (label.get('is_condition_end', False) or 
                           category1 == '工况结束' or category1 == '结束标记')
            
            # 如果当前标签不是结束标记，创建一个片段
            if not is_end_marker:
                # 确定片段的结束帧
                if i + 1 < len(temp_labels):
                    # 下一个标签的前一帧作为结束帧
                    segment_end = temp_labels[i + 1]['frame'] - 1
                else:
                    # 最后一个标签，延续到视频结束
                    segment_end = self.frame_count - 1
                
                # 将分类转换为数字编码（从左到右为0、1、2...）
                category1_code = self._category_to_code(category1, category1_options)
                category2_code = self._category_to_code(category2, category2_options) if category2 else 0
                
                # 创建片段记录
                segment = {
                    'start_frame': label_frame,
                    'end_frame': segment_end,
                    'category1_code': category1_code,
                    'category2_code': category2_code
                }
                segments.append(segment)
        
        # 每个帧区间单独占一行
        # 如果有第二个分类配置，格式为：帧区间 分类1编码 分类2编码
        # 如果没有第二个分类配置，格式为：帧区间 分类1编码
        if segments:
            has_category2 = bool(category2_options)  # 检查是否配置了第二个分类
            for segment in segments:
                if has_category2:
                    # 有第二个分类时，输出两个编码
                    lines.append(f"{segment['start_frame']}-{segment['end_frame']} {segment['category1_code']} {segment['category2_code']}")
                else:
                    # 没有第二个分类时，只输出第一个编码
                    lines.append(f"{segment['start_frame']}-{segment['end_frame']} {segment['category1_code']}")
        
        self._save_export_file(lines)

    def _category_to_code(self, category_value, category_options):
        """将分类值转换为数字编码（从左到右为0、1、2...）"""
        if not category_value or not category_options:
            return 0
        
        try:
            # 在选项列表中查找索引，从0开始
            return category_options.index(category_value)
        except ValueError:
            # 如果找不到，返回0作为默认值
            return 0

    def _save_export_file(self, lines):
        """保存导出文件"""
        default_txt_name = self.video_filename + '.txt' if self.video_filename else 'labels.txt'
        save_file, _ = QFileDialog.getSaveFileName(self, '保存标签', default_txt_name, 'Text Files (*.txt)')
        if save_file:
            with open(save_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            QMessageBox.information(self, '导出成功', '标签已成功导出。')

    def category1_to_code(self, value):
        """第一个分类值转换为代码"""
        if not self.label_config.get('category1_options'):
            return -1
        try:
            return self.label_config['category1_options'].index(value)
        except (ValueError, AttributeError):
            return -1

    def category2_to_code(self, value):
        """第二个分类值转换为代码"""
        if not self.label_config.get('category2_options'):
            return -1
        try:
            return self.label_config['category2_options'].index(value)
        except (ValueError, AttributeError):
            return -1

    def import_labels(self):
        """导入标签文件"""
        if not self.video_loaded:
            QMessageBox.warning(self, '错误', '请先加载视频文件。')
            return
        open_file, _ = QFileDialog.getOpenFileName(self, '打开标签文件', '', 'Text Files (*.txt)')
        if open_file:
            txt_filename = os.path.splitext(os.path.basename(open_file))[0]
            if txt_filename != self.video_filename:
                reply = QMessageBox.warning(self, '警告', '视频文件名和标签文件名不一致。是否继续导入？',
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return
            
            with open(open_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 检查文件格式
            if self._is_generic_label_file(lines):
                self._import_generic_labels(lines)
            else:
                self._import_boiling_labels(lines)

    def _import_generic_labels(self, lines):
        """导入通用标签"""
        # 检查是否为新的简化格式（第一行包含帧区间）
        if lines and '-' in lines[0] and ' ' in lines[0]:
            self._import_simplified_labels(lines)
            return
        
        # 解析配置信息（旧版格式）
        config_info = {}
        label_data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('# 标签配置'):
                continue
            elif line.startswith('# 第一个分类:'):
                config_info['category1_name'] = line.split(':', 1)[1].strip()
            elif line.startswith('# 第一个分类选项:'):
                config_info['category1_options'] = [opt.strip() for opt in line.split(':', 1)[1].split(',')]
            elif line.startswith('# 第二个分类:'):
                category2_name = line.split(':', 1)[1].strip()
                config_info['category2_name'] = category2_name if category2_name != '未使用' else None
            elif line.startswith('# 第二个分类选项:'):
                category2_options = line.split(':', 1)[1].strip()
                config_info['category2_options'] = [opt.strip() for opt in category2_options.split(',')] if category2_options != '未使用' else None
            elif line.startswith('#'):
                continue
            else:
                label_data_start = i
                break
        
        # 更新当前配置
        if 'category1_name' in config_info:
            self.label_config['category1_name'] = config_info['category1_name']
            self.label_config['category1_options'] = config_info['category1_options']
            self.label_config['category2_name'] = config_info.get('category2_name')
            self.label_config['category2_options'] = config_info.get('category2_options')
        
        # 解析标签数据
        temp_labels = []
        for line in lines[label_data_start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('\t')
            if len(parts) < 3:
                continue
                
            try:
                frame = int(parts[0])
                time_str = parts[1]
                category1 = parts[2]
                category2 = parts[3] if len(parts) > 3 and parts[3] else None
                
                # 如果category2为空字符串，设为None
                if category2 == '':
                    category2 = None
                
                # 检查是否为结束标记
                is_end_marker = (category1 == '结束标记' or category1 == '工况结束')
                
                label = {
                    'frame': frame,
                    'category1_value': category1,
                    'category2_value': category2,
                    'is_condition_end': is_end_marker,
                    # 保留旧字段以兼容溢锅标签
                    'boiling_state': category1,
                    'lid_type': category2 if not is_end_marker else None
                }
                temp_labels.append(label)
            except (ValueError, IndexError):
                continue
        
        # 更新标签列表
        self.labels = temp_labels
        self.labels.sort(key=lambda x: x['frame'])
        
        # 检查标签逻辑是否有效（统一使用简化的逻辑检查）
        if not self.is_label_list_valid(self.labels):
            QMessageBox.warning(self, '标签不符合逻辑', '导入的标签不符合逻辑。')
        
        self.update_label_list()
        self.progress_slider.set_labels(self.labels)
        QMessageBox.information(self, '导入成功', '标签已成功导入。')

    def _import_simplified_labels(self, lines):
        """导入简化格式的标签（第一行是帧区间和分类编码，后续行是详细标签数据）"""
        temp_labels = []
        
        # 获取当前的分类选项（用于数字编码转换）
        category1_options = self.label_config.get('category1_options', [])
        category2_options = self.label_config.get('category2_options', [])
        
        # 解析每一行的帧区间信息
        # 支持两种格式：
        # 1. 帧区间 分类1编码（没有第二个分类时）
        # 2. 帧区间 分类1编码 分类2编码（有第二个分类时）
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 2:  # 需要至少2个部分：帧区间 分类1编码
                try:
                    frame_range = parts[0]
                    category1_code = int(parts[1])
                    category2_code = int(parts[2]) if len(parts) >= 3 else 0  # 第二个分类编码（可选）
                    
                    # 解析帧区间
                    if '-' in frame_range:
                        start_frame, end_frame = frame_range.split('-')
                        start_frame = int(start_frame)
                        end_frame = int(end_frame)
                        
                        # 将数字编码转换为分类值
                        if category1_code == len(category1_options):
                            # 特殊编码表示结束标记
                            category1 = '结束标记'
                            category2 = None
                            is_end_marker = True
                        else:
                            # 正常分类编码
                            category1 = self._code_to_category(category1_code, category1_options)
                            # 修复：第二个分类编码为0时也应该被处理，0是有效的编码值
                            category2 = self._code_to_category(category2_code, category2_options) if category2_options and len(parts) >= 3 else None
                            is_end_marker = False
                        
                        # 为区间的起始帧创建标签
                        label = {
                            'frame': start_frame,
                            'endframe': end_frame,
                            'category1_value': category1,
                            'category2_value': category2,
                            'is_condition_end': is_end_marker,
                            # 保留旧字段以兼容溢锅标签
                            'boiling_state': category1,
                            'lid_type': category2 if not is_end_marker else None
                        }
                        temp_labels.append(label)
                        
                        # 如果不是结束标记，在区间结束后添加结束标记
                        if not is_end_marker and end_frame < self.frame_count - 1:
                            end_label = {
                                'frame': end_frame + 1,
                                'category1_value': '结束标记',
                                'category2_value': None,
                                'is_condition_end': True,
                                'boiling_state': '结束标记',
                                'lid_type': None
                            }
                            temp_labels.append(end_label)
                            
                except (ValueError, IndexError):
                    continue
        
        # 如果有后续行，也解析它们（兼容旧格式）
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 4:  # 需要至少4个部分：帧号 时间 分类1编码 分类2编码
                continue
                
            try:
                frame = int(parts[0])
                time_str = parts[1]
                category1_code = int(parts[2])
                category2_code = int(parts[3])
                
                # 将数字编码转换为分类值
                if category1_code == len(category1_options):
                    # 特殊编码表示结束标记
                    category1 = '结束标记'
                    category2 = None
                    is_end_marker = True
                else:
                    # 正常分类编码
                    category1 = self._code_to_category(category1_code, category1_options)
                    # 修复：第二个分类编码为0时也应该被处理，0是有效的编码值
                    category2 = self._code_to_category(category2_code, category2_options) if category2_options else None
                    is_end_marker = False
                
                label = {
                    'frame': frame,
                    'category1_value': category1,
                    'category2_value': category2,
                    'is_condition_end': is_end_marker,
                    # 保留旧字段以兼容溢锅标签
                    'boiling_state': category1,
                    'lid_type': category2 if not is_end_marker else None
                }
                temp_labels.append(label)
            except (ValueError, IndexError):
                continue
        
        # 更新标签列表
        self.labels = temp_labels
        self.labels.sort(key=lambda x: x['frame'])
        
        # 检查标签逻辑是否有效
        if not self.is_label_list_valid(self.labels):
            QMessageBox.warning(self, '标签不符合逻辑', '导入的标签不符合逻辑。')
        
        self.update_label_list()
        self.progress_slider.set_labels(self.labels)
        QMessageBox.information(self, '导入成功', '标签已成功导入。')

    def _code_to_category(self, code, category_options):
        """将数字编码转换为分类值"""
        if not category_options or code < 0 or code >= len(category_options):
            return 'Unknown'
        return category_options[code]

    def _is_generic_label_file(self, lines):
        """检查是否为通用标签文件格式"""
        for line in lines:
            if line.strip().startswith('# 标签配置'):
                return True
            # 检查是否有制表符分隔的格式（旧版通用格式）
            if '\t' in line and not line.strip().startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:  # frame, time, category1 (category2可选)
                    return True
        
        # 检查是否为新的简化格式（每行包含帧区间和分类编码）
        if lines:
            # 检查前几行是否都符合简化格式
            valid_lines = 0
            for line in lines[:3]:  # 检查前3行
                line = line.strip()
                if not line:
                    continue
                    
                # 检查是否包含帧区间格式（如：250-449 1 或 250-449 1 0）
                if '-' in line and ' ' in line:
                    parts = line.split()
                    # 检查是否为2个或3个部分的格式：帧区间 分类1编码 [分类2编码]
                    if len(parts) == 2 or len(parts) == 3:
                        try:
                            # 检查是否符合格式
                            frame_range = parts[0]
                            category1_code = int(parts[1])
                            if len(parts) == 3:
                                category2_code = int(parts[2])  # 验证第二个分类编码也是数字
                            
                            # 验证帧区间格式
                            if '-' in frame_range:
                                start, end = frame_range.split('-')
                                int(start)
                                int(end)
                                valid_lines += 1
                        except ValueError:
                            pass
            
            # 如果有有效的简化格式行，则认为是简化格式
            if valid_lines > 0:
                return True
        
        return False

    def _import_boiling_labels(self, lines):
        """导入旧版溢锅标签格式"""
        temp_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            frame_range, boiling_state_code_str, lid_type_code_str = parts
            start_frame_str, end_frame_str = frame_range.split('-')
            start_frame = int(start_frame_str)
            end_frame = int(end_frame_str)
            boiling_state_code = int(boiling_state_code_str)
            lid_type_code = int(lid_type_code_str)

            boiling_state = self.code_to_category1(boiling_state_code)
            lid_type = self.code_to_category2(lid_type_code)

            # 检查是否为结束标记
            is_end_marker = (boiling_state == '工况结束')
            
            # 使用新的标签格式
            label = {
                'frame': start_frame,
                'endframe': end_frame,
                'category1_value': '结束标记' if is_end_marker else boiling_state,
                'category2_value': lid_type if not is_end_marker else None,
                'is_condition_end': is_end_marker,
                # 保留旧字段以兼容
                'boiling_state': boiling_state,
                'lid_type': lid_type if not is_end_marker else None
            }
            temp_labels.append(label)

        # 在导入标签之后，自动插入缺失的"工况结束"
        adjusted_labels = []
        for i, label in enumerate(temp_labels):
            adjusted_labels.append(label)
            if i + 1 < len(temp_labels):
                current_label = label
                next_label = temp_labels[i + 1]

                # 检查是否不连续，若不连续插入"工况结束"
                current_state = current_label.get('category1_value') or current_label.get('boiling_state')
                next_state = next_label.get('category1_value') or next_label.get('boiling_state')
                
                if current_state != next_state and current_label['endframe'] < next_label['frame'] - 1:
                    adjusted_labels.append({
                        'frame': current_label['endframe'] + 1,
                        'category1_value': '结束标记',
                        'category2_value': None,
                        'is_condition_end': True,
                        'boiling_state': '工况结束',
                        'lid_type': None
                    })

                # 新增逻辑：若前后两个标签的状态相同，插入结束标记
                elif current_state == next_state and current_state not in ['工况结束', '结束标记']:
                    adjusted_labels.append({
                        'frame': current_label['endframe'] + 1,
                        'category1_value': '结束标记',
                        'category2_value': None,
                        'is_condition_end': True,
                        'boiling_state': '工况结束',
                        'lid_type': None
                    })

        # 更新标签列表
        self.labels = adjusted_labels
        self.labels.sort(key=lambda x: x['frame'])

        # 检查标签逻辑是否有效
        if not self.is_label_list_valid(self.labels):
            QMessageBox.warning(self, '标签不符合逻辑', '导入的标签不符合逻辑。')

        self.update_label_list()
        self.progress_slider.set_labels(self.labels)
        QMessageBox.information(self, '导入成功', '标签已成功导入。')

    def code_to_category1(self, code):
        """代码转换为第一个分类值"""
        if not self.label_config.get('category1_options') or code < 0:
            return 'Unknown'
        try:
            return self.label_config['category1_options'][code]
        except (IndexError, TypeError):
            return 'Unknown'

    def code_to_category2(self, code):
        """代码转换为第二个分类值"""
        if not self.label_config.get('category2_options') or code < 0:
            return 'Unknown'
        try:
            return self.label_config['category2_options'][code]
        except (IndexError, TypeError):
            return 'Unknown'

    def update_progress_slider_labels(self):
        self.progress_slider.set_labels(self.labels)

    def handle_left_right_keys(self, frames):
        selected_items = self.label_list.selectedItems()
        if selected_items:
            self.move_selected_label(frames)
        else:
            self.seek_frames(frames)

    def move_selected_label(self, frames):
        selected_items = self.label_list.selectedItems()
        if selected_items:
            index = self.label_list.row(selected_items[0])
            label = self.labels[index]
            new_frame = label['frame'] + frames
            if 0 <= new_frame < self.frame_count:
                temp_labels = copy.deepcopy(self.labels)
                temp_labels[index]['frame'] = new_frame
                temp_labels.sort(key=lambda x: x['frame'])
                if self.is_label_list_valid(temp_labels):
                    self.labels = temp_labels
                    self.update_label_list()
                    self.progress_slider.set_labels(self.labels)
                    for i, lbl in enumerate(self.labels):
                        if lbl == temp_labels[index]:
                            self.label_list.setCurrentRow(i)
                            self.current_frame = self.labels[i]['frame']
                            self.display_frame()
                            break
                else:
                    QMessageBox.warning(self, '无效的标签位置', '移动标签后标签逻辑无效，无法移动。')
            else:
                QMessageBox.warning(self, '无效的标签位置', '标签位置超出视频帧范围。')

    def open_frame_extraction_dialog(self):
        """打开拆帧对话框 - 独立拆帧功能，无需加载视频或标签"""
        # 直接打开通用拆帧对话框，不再检查视频加载状态和标签
        # 这样用户可以独立使用拆帧功能，处理任意视频文件
        dialog = UniversalFrameExtractionDialog(self)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoLabelingTool()
    window.showMaximized()
    sys.exit(app.exec_())
