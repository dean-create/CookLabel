#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import os.path
import platform
import shutil
import sys
import time
import webbrowser as wb
from functools import partial

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.lightWidget import LightWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError, LabelFileFormat
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.create_ml_io import CreateMLReader
from libs.create_ml_io import JSON_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
from libs.cropDialog import CropDialog
from libs.modelInferenceDialog import ModelInferenceDialog
from libs.nanodetInferenceDialog import NanoDetInferenceDialog
from libs.videoFrameExtractor import VideoFrameExtractorDialog

__appname__ = 'CookLabel'


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            add_actions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            add_actions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, default_filename=None, default_prefdef_class_file=None, default_save_dir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        
        # 设置焦点策略，确保主窗口能接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        self.os_name = platform.system()

        # Load string bundle for i18n
        self.string_bundle = StringBundle.get_bundle()
        get_str = lambda str_id: self.string_bundle.get_string(str_id)

        # Save as Pascal voc xml
        self.default_save_dir = default_save_dir
        self.label_file_format = settings.get(SETTING_LABEL_FILE_FORMAT, LabelFileFormat.PASCAL_VOC)

        # For loading all image under a directory
        self.m_img_list = []
        self.dir_name = None
        self.label_hist = []
        self.last_open_dir = None
        self.cur_img_idx = 0
        self.img_count = len(self.m_img_list)

        # 进度保存相关属性
        self.progress_dir = os.path.join(os.getcwd(), 'progress')  # 进度文件夹路径
        self.progress_file = os.path.join(self.progress_dir, 'progress.txt')  # 进度文件路径
        self.current_project_dir = None  # 当前项目目录路径

        # Whether we need to save or not.
        self.dirty = False

        self._no_selection_slot = False
        self._beginner = True
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.load_predefined_classes(default_prefdef_class_file)

        if not self.label_hist:
            # print("Not find:/data/predefined_classes.txt (optional)")
            pass

        # Main widgets and related state.
        # 为LabelDialog提供默认的标签配置，保持与原有功能的兼容性
        self.label_dialog = LabelDialog(parent=self, list_item=self.label_hist)

        self.items_to_shapes = {}
        self.shapes_to_items = {}
        self.prev_label_text = ''

        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for edit button
        self.edit_button = QToolButton()
        self.edit_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add edit button to list_layout
        list_layout.addWidget(self.edit_button)

        # Create and add combobox for showing unique labels in group
        # self.combo_box = ComboBox(self)
        # list_layout.addWidget(self.combo_box)

        # Create and add a widget for showing current label items
        self.label_list = QListWidget()
        label_list_container = QWidget()
        label_list_container.setLayout(list_layout)
        self.label_list.itemActivated.connect(self.label_selection_changed)
        self.label_list.itemSelectionChanged.connect(self.label_selection_changed)
        self.label_list.itemDoubleClicked.connect(self.edit_label)
        # Connect to itemChanged to detect checkbox changes.
        self.label_list.itemChanged.connect(self.label_item_changed)
        list_layout.addWidget(self.label_list)



        self.dock = QDockWidget(get_str('boxLabelText'), self)
        self.dock.setObjectName(get_str('labels'))
        self.dock.setWidget(label_list_container)

        self.file_list_widget = QListWidget()
        self.file_list_widget.itemDoubleClicked.connect(self.file_item_double_clicked)
        file_list_layout = QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.addWidget(self.file_list_widget)
        file_list_container = QWidget()
        file_list_container.setLayout(file_list_layout)
        self.file_dock = QDockWidget(get_str('fileList'), self)
        self.file_dock.setObjectName(get_str('files'))
        self.file_dock.setWidget(file_list_container)

        self.zoom_widget = ZoomWidget()
        self.light_widget = LightWidget(get_str('lightWidgetTitle'))
        self.color_dialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoom_request)
        self.canvas.lightRequest.connect(self.light_request)
        self.canvas.set_drawing_shape_to_square(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scroll_area = scroll
        self.canvas.scrollRequest.connect(self.scroll_request)

        self.canvas.newShape.connect(self.new_shape)
        self.canvas.shapeMoved.connect(self.set_dirty)
        self.canvas.selectionChanged.connect(self.shape_selection_changed)
        self.canvas.drawingPolygon.connect(self.toggle_drawing_sensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.file_dock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dock_features = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dock_features)

        # Actions
        action = partial(new_action, self)
        quit = action(get_str('quit'), self.close,
                      'Ctrl+Q', 'quit', get_str('quitApp'))

        open = action(get_str('openFile'), self.open_file,
                      'Ctrl+O', 'modellabel', get_str('openFileDetail'))

        open_dir = action(get_str('openDir'), self.open_dir_dialog,
                          'Ctrl+u', 'opennew', get_str('openDir'))

        change_save_dir = action(get_str('changeSaveDir'), self.change_save_dir_dialog,
                                 'Ctrl+r', 'opennew', get_str('changeSaveDir'))

        # 移除开启标签功能 - 注释掉以下代码
        # open_annotation = action(get_str('openAnnotation'), self.open_annotation_dialog,
        #                          'Ctrl+Shift+O', 'opennew', get_str('openAnnotationDetail'))
        copy_prev_bounding = action(get_str('copyPrevBounding'), self.copy_previous_bounding_boxes, 'Ctrl+v', 'copy', get_str('copyPrevBounding'))

        open_next_image = action(get_str('nextImg'), self.open_next_image,
                                 'd', 'next', get_str('nextImgDetail'))

        open_prev_image = action(get_str('prevImg'), self.open_prev_image,
                                 'a', 'prev', get_str('prevImgDetail'))

        verify = action(get_str('verifyImg'), self.verify_image,
                        'space', 'resize', get_str('verifyImgDetail'))

        save = action(get_str('save'), self.save_file,
                      'Ctrl+S', 'baocun', get_str('saveDetail'), enabled=False)

        def get_format_meta(format):
            """
            returns a tuple containing (title, icon_name) of the selected format
            """
            if format == LabelFileFormat.PASCAL_VOC:
                return '&PascalVOC', 'format_voc'
            elif format == LabelFileFormat.YOLO:
                return '&YOLO', 'format_yolo'
            elif format == LabelFileFormat.CREATE_ML:
                return '&CreateML', 'format_createml'

        save_format = action(get_format_meta(self.label_file_format)[0],
                             self.change_format, 'Ctrl+Y',
                             get_format_meta(self.label_file_format)[1],
                             get_str('changeSaveFormat'), enabled=True)

        save_as = action(get_str('saveAs'), self.save_file_as,
                         'Ctrl+Shift+S', 'save-as', get_str('saveAsDetail'), enabled=False)

        close = action(get_str('closeCur'), self.close_file, 'Ctrl+W', 'close', get_str('closeCurDetail'))

        delete_image = action(get_str('deleteImg'), self.delete_image, 'E', 'close', get_str('deleteImgDetail'))

        # restore_image = action('恢复图片', self.restore_image, None, 'undo', '恢复上一张移除的图片')  # 移除恢复图片按钮，因为快捷键Q的撤回操作已实现相同功能
        restore_last_operation = action('撤回操作', self.restore_last_operation, 'Q', 'undo', '撤回上一次删除或分类操作')

        reset_all = action(get_str('resetAll'), self.reset_all, None, 'resetall', get_str('resetAllDetail'))

        color1 = action(get_str('boxLineColor'), self.choose_color1,
                        'Ctrl+L', 'color_line', get_str('boxLineColorDetail'))

        create_mode = action(get_str('crtBox'), self.set_create_mode,
                             'w', 'new', get_str('crtBoxDetail'), enabled=False)
        edit_mode = action(get_str('editBox'), self.set_edit_mode,
                           'Ctrl+J', 'edit', get_str('editBoxDetail'), enabled=False)

        create = action(get_str('crtBox'), self.create_shape,
                        'w', 'new', get_str('crtBoxDetail'), enabled=False)
        delete = action(get_str('delBox'), self.delete_selected_shape,
                        'Delete', 'delete11', get_str('delBoxDetail'), enabled=False)
        copy = action(get_str('dupBox'), self.copy_selected_shape,
                      'Ctrl+D', 'xinfuzhi', get_str('dupBoxDetail'),
                      enabled=False)

        advanced_mode = action(get_str('advancedMode'), self.toggle_advanced_mode,
                               'Ctrl+Shift+A', 'expert', get_str('advancedModeDetail'),
                               checkable=True)

        hide_all = action(get_str('hideAllBox'), partial(self.toggle_polygons, False),
                          'Ctrl+H', 'hide', get_str('hideAllBoxDetail'),
                          enabled=False)
        show_all = action(get_str('showAllBox'), partial(self.toggle_polygons, True),
                          'Ctrl+A', 'hide', get_str('showAllBoxDetail'),
                          enabled=False)

        help_default = action(get_str('tutorialDefault'), self.show_default_tutorial_dialog, None, 'help', get_str('tutorialDetail'))
        show_info = action(get_str('info'), self.show_info_dialog, None, 'help', get_str('info'))
        show_shortcut = action(get_str('shortcut'), self.show_shortcuts_dialog, None, 'help', get_str('shortcut'))
        show_description = action(get_str('description'), self.show_description_dialog, None, 'help', get_str('description'))
        
        # 设置菜单动作定义
        theme_color = action(get_str('themeColor'), self.change_theme_color, None, 'boxcolor', get_str('themeColorDetail'))
        annotation_box_color = action(get_str('annotationBoxColor'), self.change_annotation_box_color, None, 'boxcolor', get_str('annotationBoxColorDetail'))

        # 视频拆帧功能动作定义
        video_frame_fixed = action(get_str('videoFrameFixed'), self.open_video_frame_fixed, 
                                   'Ctrl+T', 'framev', get_str('videoFrameFixedDetail'))
        video_frame_uniform = action(get_str('videoFrameUniform'), self.open_video_frame_uniform, 
                                     'Ctrl+U', 'framev', get_str('videoFrameUniformDetail'))
        video_frame_tracking = action(get_str('videoFrameTracking'), self.open_video_frame_tracking, 
                                      'Ctrl+Z', 'framev', get_str('videoFrameTrackingDetail'))
        video_frame_manual = action(get_str('videoFrameManual'), self.open_video_frame_manual, 
                                    'Ctrl+M', 'framev', get_str('videoFrameManualDetail'))

        # 数据清洗功能动作定义
        data_cleaning_duplicate = action(get_str('dataCleaningDuplicate'), self.data_cleaning_duplicate, 
                                         None, 'data_clean', get_str('dataCleaningDuplicateDetail'))
        data_cleaning_blur = action(get_str('dataCleaningBlur'), self.data_cleaning_blur, 
                                   None, 'data_clean', get_str('dataCleaningBlurDetail'))
        data_cleaning_overexposure = action(get_str('dataCleaningOverexposure'), self.data_cleaning_overexposure, 
                                           None, 'data_clean', get_str('dataCleaningOverexposureDetail'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoom_widget)
        self.zoom_widget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (format_shortcut("Ctrl+[-+]"),
                                             format_shortcut("Ctrl+Wheel")))
        self.zoom_widget.setEnabled(False)

        zoom_in = action(get_str('zoomin'), partial(self.add_zoom, 10),
                         'Ctrl++', 'zoom-in', get_str('zoominDetail'), enabled=False)
        zoom_out = action(get_str('zoomout'), partial(self.add_zoom, -10),
                          'Ctrl+-', 'zoom-out', get_str('zoomoutDetail'), enabled=False)
        zoom_org = action(get_str('originalsize'), partial(self.set_zoom, 100),
                          'Ctrl+=', 'zoom', get_str('originalsizeDetail'), enabled=False)
        fit_window = action(get_str('fitWin'), self.set_fit_window,
                            'Ctrl+F', 'fit-window', get_str('fitWinDetail'),
                            checkable=True, enabled=False)
        fit_width = action(get_str('fitWidth'), self.set_fit_width,
                           'Ctrl+Shift+F', 'fit-width', get_str('fitWidthDetail'),
                           checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoom_actions = (self.zoom_widget, zoom_in, zoom_out,
                        zoom_org, fit_window, fit_width)
        self.zoom_mode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            self.FIT_WIDTH: self.scale_fit_width,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        light = QWidgetAction(self)
        light.setDefaultWidget(self.light_widget)
        self.light_widget.setWhatsThis(
            u"Brighten or darken current image. Also accessible with"
            " %s and %s from the canvas." % (format_shortcut("Ctrl+Shift+[-+]"),
                                             format_shortcut("Ctrl+Shift+Wheel")))
        self.light_widget.setEnabled(False)

        light_brighten = action(get_str('lightbrighten'), partial(self.add_light, 10),
                                'Ctrl+Up', 'light_lighten', get_str('lightbrightenDetail'), enabled=False)
        light_darken = action(get_str('lightdarken'), partial(self.add_light, -10),
                              'Ctrl+Down', 'light_darken', get_str('lightdarkenDetail'), enabled=False)
        light_org = action(get_str('lightreset'), partial(self.set_light, 50),
                           'S', 'light_reset', get_str('lightresetDetail'), checkable=True, enabled=False)
        light_org.setChecked(True)

        # Group light controls into a list for easier toggling.
        light_actions = (self.light_widget, light_brighten,
                         light_darken, light_org)

        edit = action(get_str('editLabel'), self.edit_label,
                      'Ctrl+E', 'edit', get_str('editLabelDetail'),
                      enabled=False)
        self.edit_button.setDefaultAction(edit)

        shape_line_color = action(get_str('shapeLineColor'), self.choose_shape_line_color,
                                  icon='color_line', tip=get_str('shapeLineColorDetail'),
                                  enabled=False)
        shape_fill_color = action(get_str('shapeFillColor'), self.choose_shape_fill_color,
                                  icon='color', tip=get_str('shapeFillColorDetail'),
                                  enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(get_str('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        label_menu = QMenu()
        add_actions(label_menu, (edit, delete))
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(
            self.pop_label_list_menu)

        # Draw squares/rectangles
        self.draw_squares_option = QAction(get_str('drawSquares'), self)
        self.draw_squares_option.setShortcut('Ctrl+Shift+R')
        self.draw_squares_option.setCheckable(True)
        self.draw_squares_option.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.draw_squares_option.triggered.connect(self.toggle_draw_square)

        # Store actions for further handling.
        self.actions = Struct(save=save, save_format=save_format, saveAs=save_as, open=open, close=close, resetAll=reset_all, deleteImg=delete_image, # restoreImg=restore_image,  # 移除恢复图片按钮引用
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=create_mode, editMode=edit_mode, advancedMode=advanced_mode,
                              shapeLineColor=shape_line_color, shapeFillColor=shape_fill_color,
                              zoom=zoom, zoomIn=zoom_in, zoomOut=zoom_out, zoomOrg=zoom_org,
                              fitWindow=fit_window, fitWidth=fit_width,
                              zoomActions=zoom_actions,
                              lightBrighten=light_brighten, lightDarken=light_darken, lightOrg=light_org,
                              lightActions=light_actions,
                              fileMenuActions=(
                                  open, open_dir, save, save_as, close, reset_all, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.draw_squares_option),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(create_mode, edit_mode, edit, copy,
                                               delete, shape_line_color, shape_fill_color),
                              onLoadActive=(
                                  close, create, create_mode, edit_mode),
                              onShapesPresent=(save_as, hide_all, show_all))

        self.menus = Struct(
            file=self.menu(get_str('menu_file')),
            edit=self.menu(get_str('menu_edit')),
            view=self.menu(get_str('menu_view')),
            video=self.menu(get_str('menu_video')),
            datacleaning=self.menu(get_str('menu_datacleaning')),
            help=self.menu(get_str('menu_help')),
            settings=self.menu(get_str('menu_settings')),  # 添加设置菜单
            # 移除最近打开菜单功能
            # recentFiles=QMenu(get_str('menu_openRecent')),
            labelList=label_menu)

        # Auto saving : Enable auto saving if pressing next
        self.auto_saving = QAction(get_str('autoSaveMode'), self)
        self.auto_saving.setCheckable(True)
        self.auto_saving.setChecked(settings.get(SETTING_AUTO_SAVE, True))  # 默认开启自动保存
        # Sync single class mode from PR#106
        self.single_class_mode = QAction(get_str('singleClsMode'), self)
        self.single_class_mode.setShortcut("Ctrl+Shift+S")
        self.single_class_mode.setCheckable(True)
        self.single_class_mode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.display_label_option = QAction(get_str('displayLabel'), self)
        self.display_label_option.setShortcut("Ctrl+Shift+P")
        self.display_label_option.setCheckable(True)
        self.display_label_option.setChecked(settings.get(SETTING_PAINT_LABEL, True))  # 默认开启显示类别
        self.display_label_option.triggered.connect(self.toggle_paint_labels_option)

        add_actions(self.menus.file,
                    (open, open_dir, change_save_dir, copy_prev_bounding, save, save_format, save_as, close, reset_all, delete_image, restore_last_operation, quit))  # 移除restore_image，因为快捷键Q已实现相同功能
        # 视频拆帧菜单：四个子功能
        add_actions(self.menus.video, (video_frame_fixed, video_frame_uniform, video_frame_tracking, video_frame_manual))
        # 数据清洗菜单：三个子功能
        add_actions(self.menus.datacleaning, (data_cleaning_duplicate, data_cleaning_blur, data_cleaning_overexposure))
        # 帮助菜单：版本信息、快捷键、说明
        add_actions(self.menus.help, (show_info, show_shortcut, show_description))
        # 设置菜单：主题颜色、标注框颜色
        add_actions(self.menus.settings, (theme_color, annotation_box_color))
        add_actions(self.menus.view, (
            self.auto_saving,
            self.single_class_mode,
            self.display_label_option,
            labels, advanced_mode, None,
            hide_all, show_all, None,
            zoom_in, zoom_out, zoom_org, None,
            fit_window, fit_width, None,
            light_brighten, light_darken, light_org))

        # 移除最近打开功能，注释掉文件菜单更新连接
        # self.menus.file.aboutToShow.connect(self.update_file_menu)

        # Custom context menu for the canvas widget:
        add_actions(self.canvas.menus[0], self.actions.beginnerContext)
        add_actions(self.canvas.menus[1], (
            action('&Copy here', self.copy_shape),
            action('&Move here', self.move_shape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, open_dir, change_save_dir, open_next_image, open_prev_image, verify, save, save_format, None, create, copy, delete, None,
            zoom_in, zoom, zoom_out, fit_window, fit_width, None,
            light_brighten, light, light_darken, light_org)

        self.actions.advanced = (
            open, open_dir, change_save_dir, open_next_image, open_prev_image, save, save_format, None,
            create_mode, edit_mode, None,
            hide_all, show_all)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.file_path = ustr(default_filename)
        self.last_open_dir = None
        self.recent_files = []
        self.max_recent = 7
        self.line_color = None
        self.fill_color = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris - difficult feature removed, keeping default value
        self.difficult = False

        # Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recent_file_qstring_list = settings.get(SETTING_RECENT_FILES)
                self.recent_files = [ustr(i) for i in recent_file_qstring_list]
            else:
                self.recent_files = recent_file_qstring_list = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        save_dir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.last_open_dir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.default_save_dir is None and save_dir is not None and os.path.exists(save_dir):
            self.default_save_dir = save_dir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.default_save_dir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.line_color = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fill_color = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.set_drawing_color(self.line_color)
        # Add chris - Shape.difficult setting removed as UI control was removed

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggle_advanced_mode()

        # 移除最近打开功能，注释掉文件菜单动态填充
        # Populate the File menu dynamically.
        # self.update_file_menu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.file_path and os.path.isdir(self.file_path):
            self.queue_event(partial(self.import_dir_images, self.file_path or ""))
        elif self.file_path:
            self.queue_event(partial(self.load_file, self.file_path or ""))

        # Callbacks:
        self.zoom_widget.valueChanged.connect(self.paint_canvas)
        self.light_widget.valueChanged.connect(self.paint_canvas)

        self.populate_mode_actions()

        # Display cursor coordinates at the right of status bar
        self.label_coordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.label_coordinates)

        # Open Dir if default file
        if self.file_path and os.path.isdir(self.file_path):
            self.open_dir_dialog(dir_path=self.file_path, silent=True)
        
        # 初始化进度记录相关变量
        self.progress_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress')
        self.progress_file = os.path.join(self.progress_dir, 'progress.txt')
        self.current_project_dir = None  # 当前项目目录路径
        
        # 初始化删除图片跟踪变量
        self.deleted_images = []  # 存储删除的图片信息：[(原路径, Delete文件夹路径), ...]
        
        # 初始化分类操作跟踪变量
        self.classified_images = []  # 存储分类的图片信息：[{原路径, 分类文件夹路径, 原图片索引, 标签文件信息}, ...]
        
        # 初始化操作历史记录，用于统一撤回功能
        self.operation_history = []  # 存储所有操作的历史记录：[{'type': 'delete'/'classify', 'timestamp': time, 'data': operation_data}, ...]
        
        # 添加标志位，用于区分是否在撤回操作中，避免触发多余的恢复对话框
        self.is_restoring_operation = False
        
        # 强制设置自动保存和显示类别功能为默认勾选状态
        # 这样可以确保无论配置文件中的设置如何，这两个功能都会默认启用
        self.auto_saving.setChecked(True)  # 强制启用自动保存功能
        self.display_label_option.setChecked(True)  # 强制启用显示类别功能
        
        # 加载用户设置
        self.load_settings()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.set_drawing_shape_to_square(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.set_drawing_shape_to_square(True)
        elif event.key() == Qt.Key_Up and event.modifiers() == Qt.ControlModifier:
            # Ctrl+上方向键：提亮图片
            self.add_light(10)
        elif event.key() == Qt.Key_Down and event.modifiers() == Qt.ControlModifier:
            # Ctrl+下方向键：变暗图片
            self.add_light(-10)
        elif event.key() == Qt.Key_Up:
            # 上方向键：向上微调标注框
            if self.canvas.selected_shape:
                self.canvas.move_one_pixel('Up')
            else:
                # 如果没有选中标注框，将事件传递给canvas处理
                self.canvas.keyPressEvent(event)
        elif event.key() == Qt.Key_Down:
            # 下方向键：向下微调标注框
            if self.canvas.selected_shape:
                self.canvas.move_one_pixel('Down')
            else:
                # 如果没有选中标注框，将事件传递给canvas处理
                self.canvas.keyPressEvent(event)
        elif event.key() == Qt.Key_S:
            # 恢复原始亮度
            self.set_light(50)
        elif event.key() == Qt.Key_0:
            # 将当前图片分类到类别0
            self.classify_image(0)
        elif event.key() == Qt.Key_1:
            # 将当前图片分类到类别1
            self.classify_image(1)
        elif event.key() == Qt.Key_2:
            # 将当前图片分类到类别2
            self.classify_image(2)
        elif event.key() == Qt.Key_3:
            # 将当前图片分类到类别3
            self.classify_image(3)
        elif event.key() == Qt.Key_4:
            # 将当前图片分类到类别4
            self.classify_image(4)

    # Support Functions #
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(new_icon("format_voc"))
            self.label_file_format = LabelFileFormat.PASCAL_VOC
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(new_icon("format_yolo"))
            self.label_file_format = LabelFileFormat.YOLO
            LabelFile.suffix = TXT_EXT

        elif save_format == FORMAT_CREATEML:
            self.actions.save_format.setText(FORMAT_CREATEML)
            self.actions.save_format.setIcon(new_icon("format_createml"))
            self.label_file_format = LabelFileFormat.CREATE_ML
            LabelFile.suffix = JSON_EXT

    def change_format(self):
        if self.label_file_format == LabelFileFormat.PASCAL_VOC:
            self.set_format(FORMAT_YOLO)
        elif self.label_file_format == LabelFileFormat.YOLO:
            self.set_format(FORMAT_CREATEML)
        elif self.label_file_format == LabelFileFormat.CREATE_ML:
            self.set_format(FORMAT_PASCALVOC)
        else:
            raise ValueError('Unknown label file format.')
        self.set_dirty()

    def no_shapes(self):
        return not self.items_to_shapes

    def toggle_advanced_mode(self, value=True):
        self._beginner = not value
        self.canvas.set_editing(True)
        self.populate_mode_actions()
        self.edit_button.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dock_features)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dock_features)

    def populate_mode_actions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        add_actions(self.tools, tool)
        self.canvas.menus[0].clear()
        add_actions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        add_actions(self.menus.edit, actions + self.actions.editMenu)

    def set_beginner(self):
        self.tools.clear()
        add_actions(self.tools, self.actions.beginner)

    def set_advanced(self):
        self.tools.clear()
        add_actions(self.tools, self.actions.advanced)

    def set_dirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def set_clean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggle_actions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for z in self.actions.lightActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queue_event(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def reset_state(self):
        self.items_to_shapes.clear()
        self.shapes_to_items.clear()
        self.label_list.clear()
        self.file_path = None
        self.image_data = None
        self.label_file = None
        self.canvas.reset_state()
        self.label_coordinates.clear()
        # self.combo_box.cb.clear()  # 注释掉组合框清空操作

    def current_item(self):
        items = self.label_list.selectedItems()
        if items:
            return items[0]
        return None

    def add_recent_file(self, file_path):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        elif len(self.recent_files) >= self.max_recent:
            self.recent_files.pop()
        self.recent_files.insert(0, file_path)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def show_tutorial_dialog(self, browser='default', link=None):
        if link is None:
            link = self.screencast

        if browser.lower() == 'default':
            wb.open(link, new=2)
        elif browser.lower() == 'chrome' and self.os_name == 'Windows':
            if shutil.which(browser.lower()):  # 'chrome' not in wb._browsers in windows
                wb.register('chrome', None, wb.BackgroundBrowser('chrome'))
            else:
                chrome_path="D:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
                if os.path.isfile(chrome_path):
                    wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))
            try:
                wb.get('chrome').open(link, new=2)
            except:
                wb.open(link, new=2)
        elif browser.lower() in wb._browsers:
            wb.get(browser.lower()).open(link, new=2)

    def show_default_tutorial_dialog(self):
        self.show_tutorial_dialog(browser='default')

    def show_info_dialog(self):
        """显示程序信息对话框"""
        from libs.__init__ import __version__
        # 只显示程序名称和版本号，简化信息显示
        msg = u'Name: {0}\nApp Version: {1}'.format(__appname__, __version__)
        QMessageBox.information(self, u'Information', msg)

    def show_description_dialog(self):
        """显示软件功能说明对话框"""
        from libs.__init__ import __version__
        
        # 创建说明对话框
        description_dialog = QDialog(self)
        description_dialog.setWindowTitle("软件说明")
        description_dialog.setMinimumSize(700, 600)
        description_dialog.setModal(True)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 软件功能说明内容
        description_info = f"""
<h2 style="color: #2E86AB; text-align: center;">CookLabelv{__version__}</h2>
<h3 style="color: #A23B72; text-align: center;">基于LabelImg增强的图像标注工具</h3>

<div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #2E86AB;">
<h3 style="color: #2E86AB; margin-top: 0;">🎯 软件简介</h3>
<p style="line-height: 1.6;">
CookLabel是基于开源项目LabelImg开发的增强版图像标注工具，在保留原有标注功能的基础上，新增了多项实用功能，实现从采集的原始数据到模型的训练数据工作流程的一体化。
</p>
</div>

<div style="margin: 20px 0; padding: 15px; background-color: #f0f8ff; border-left: 4px solid #A23B72;">
<h3 style="color: #A23B72; margin-top: 0;">✨ 新增功能特性</h3>
<ul style="line-height: 1.8;">
<li><strong>进度记录功能：</strong>自动记录标注进度，退出后可恢复到上次标注位置</li>
<li><strong>图片分类功能：</strong>支持数字键0-4快速分类图片到不同文件夹</li>
<li><strong>图片删除与恢复：</strong>支持删除图片并可一键恢复误删图片</li>
<li><strong>图片裁剪功能：</strong>内置图片裁剪工具，方便预处理</li>
<li><strong>亮度调节功能：</strong>支持实时调节图片亮度，便于标注暗图</li>
<li><strong>增强的快捷键：</strong>优化快捷键布局，提高操作效率</li>
<li><strong>智能文件管理：</strong>分离图片和标签文件夹选择，避免误操作</li>
</ul>
</div>

<div style="margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-left: 4px solid #28a745;">
<h3 style="color: #28a745; margin-top: 0;">🔧 核心功能</h3>
<ul style="line-height: 1.8;">
<li><strong>矩形标注：</strong>支持创建、编辑、删除矩形标注框</li>
<li><strong>多格式支持：</strong>支持YOLO、Pascal VOC、CreateML等多种标注格式</li>
<li><strong>标签管理：</strong>支持自定义标签类别，颜色管理</li>
<li><strong>批量操作：</strong>支持批量标注、复制粘贴标注框</li>
<li><strong>视图控制：</strong>支持缩放、适应窗口、全屏等视图操作</li>
<li><strong>自动保存：</strong>支持自动保存标注结果，防止数据丢失</li>
</ul>
</div>

<div style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107;">
<h3 style="color: #856404; margin-top: 0;">💡 使用建议</h3>
<ul style="line-height: 1.8;">
<li>首次使用建议先查看快捷键说明，熟悉操作方式</li>
<li>开启自动保存功能，避免标注数据丢失</li>
<li>合理使用图片分类功能，提高数据整理效率</li>
<li>利用进度记录功能，支持大批量标注任务的分段完成</li>
<li>使用亮度调节功能处理光线不佳的图片</li>
</ul>
</div>

<div style="margin: 20px 0; padding: 15px; background-color: #e7f3ff; border-left: 4px solid #0066cc;">
<h3 style="color: #0066cc; margin-top: 0;">📝 版权说明</h3>
<p style="line-height: 1.6;">
本软件基于开源项目LabelImg进行功能增强开发，遵循原项目的开源协议。感谢该项目的贡献者们为计算机视觉社区提供的优秀工具！
</p>
</div>
"""
        
        # 创建文本标签显示说明信息
        info_label = QLabel(description_info)
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setAlignment(Qt.AlignTop)
        info_label.setStyleSheet("QLabel { padding: 10px; }")
        
        scroll_layout.addWidget(info_label)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        
        # 添加关闭按钮
        button_layout = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(description_dialog.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        
        # 设置主布局
        layout.addWidget(scroll_area)
        layout.addLayout(button_layout)
        description_dialog.setLayout(layout)
        
        # 显示对话框
        description_dialog.exec_()

    # 进度记录相关方法
    def save_progress(self):
        """保存当前标注进度到当前图片文件夹中"""
        try:
            # 只有在有项目目录时才保存进度
            if self.current_project_dir and self.img_count > 0:
                # 将进度文件保存到当前图片文件夹中
                progress_file_path = os.path.join(self.current_project_dir, 'progress.txt')
                
                with open(progress_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"{self.cur_img_idx}\n")  # 当前图片索引
                    f.write(f"{self.current_project_dir}\n")  # 项目目录路径
                    f.write(f"{self.img_count}\n")  # 总图片数量
                
        except Exception as e:
            print(f"保存进度时出错: {e}")

    def auto_delete_progress_file(self):
        """当查看到最后一张图片时自动删除progress.txt文件"""
        try:
            if self.current_project_dir:
                progress_file_path = os.path.join(self.current_project_dir, 'progress.txt')
                if os.path.exists(progress_file_path):
                    os.remove(progress_file_path)
                    print("已自动删除progress.txt文件（已查看完所有图片）")
        except Exception as e:
            print(f"删除progress.txt文件时出错: {e}")

    def load_progress(self):
        """从当前图片文件夹中加载标注进度"""
        try:
            # 如果有当前项目目录，尝试从该目录加载进度文件
            if self.current_project_dir:
                progress_file_path = os.path.join(self.current_project_dir, 'progress.txt')
                if os.path.exists(progress_file_path):
                    with open(progress_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= 3:
                            current_index = int(lines[0].strip())
                            project_dir = lines[1].strip()
                            total_images = int(lines[2].strip())
                            return {
                                'current_index': current_index,
                                'project_dir': project_dir,
                                'total_images': total_images
                            }
        except Exception as e:
            print(f"加载进度时出错: {e}")
        return None

    def ask_restore_progress(self, progress_data):
        """询问用户是否恢复到上次的标注进度"""
        current_index = progress_data['current_index']
        total_images = progress_data['total_images']
        project_dir = progress_data['project_dir']
        
        # 创建自定义对话框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("恢复标注进度")
        msg_box.setText(f"检测到上次标注进度：\n\n"
                       f"项目目录: {os.path.basename(project_dir)}\n"
                       f"上次标注到第 {current_index + 1} 张图片\n"
                       f"总共 {total_images} 张图片\n\n"
                       f"是否要恢复到上次的标注位置？")
        
        # 添加自定义按钮
        yes_button = msg_box.addButton("是，恢复进度", QMessageBox.YesRole)
        no_button = msg_box.addButton("否，从头开始", QMessageBox.NoRole)
        
        msg_box.setDefaultButton(yes_button)
        msg_box.exec_()
        
        # 返回用户的选择
        return msg_box.clickedButton() == yes_button

    def restore_progress(self, target_index):
        """恢复到指定的图片索引位置"""
        if 0 <= target_index < self.img_count:
            self.cur_img_idx = target_index
            filename = self.m_img_list[self.cur_img_idx]
            if filename:
                self.load_file(filename, auto_load_annotations=False)
                self.statusBar().showMessage(f'已恢复到第 {target_index + 1} 张图片', 3000)
                return True
        return False

    def restore_progress_without_auto_save(self, target_index):
        """恢复到指定的图片索引位置，但不自动触发标签目录选择"""
        if 0 <= target_index < self.img_count:
            self.cur_img_idx = target_index
            filename = self.m_img_list[self.cur_img_idx]
            if filename:
                # 临时禁用自动保存，避免触发标签目录选择对话框
                original_auto_save = self.auto_saving.isChecked()
                self.auto_saving.setChecked(False)
                
                self.load_file(filename, auto_load_annotations=False)
                self.statusBar().showMessage(f'已恢复到第 {target_index + 1} 张图片', 3000)
                
                # 恢复原始的自动保存设置
                self.auto_saving.setChecked(original_auto_save)
                
                # 保存当前进度
                self.save_progress()
                return True
        return False

    def open_next_image_without_auto_save(self, _value=False):
        """打开下一张图片，但不自动触发标签目录选择"""
        if not self.may_continue():
            return

        if self.img_count <= 0:
            return
        
        if not self.m_img_list:
            return

        filename = None
        if self.file_path is None:
            filename = self.m_img_list[0]
            self.cur_img_idx = 0
        else:
            if self.cur_img_idx + 1 < self.img_count:
                self.cur_img_idx += 1
                filename = self.m_img_list[self.cur_img_idx]

        if filename:
            # 临时禁用自动保存，避免触发标签目录选择对话框
            original_auto_save = self.auto_saving.isChecked()
            self.auto_saving.setChecked(False)
            
            self.load_file(filename, auto_load_annotations=False)
            
            # 恢复原始的自动保存设置
            self.auto_saving.setChecked(original_auto_save)
            
            # 保存当前进度
            self.save_progress()
            
            # 检查是否已经查看到最后一张图片
            if self.cur_img_idx == self.img_count - 1:
                # 自动删除progress.txt文件
                self.auto_delete_progress_file()

    def show_shortcuts_dialog(self):
        """显示快捷键帮助窗口"""
        # 创建快捷键信息对话框
        shortcuts_dialog = QDialog(self)
        shortcuts_dialog.setWindowTitle("快捷键帮助")
        shortcuts_dialog.setMinimumSize(600, 500)
        shortcuts_dialog.setModal(True)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 获取版本信息
        from libs.__init__ import __version__
        
        # 快捷键信息
        shortcuts_info = f"""
<h2 style="color: #2E86AB;">CookLabelv{__version__}快捷键帮助</h2>

<h3 style="color: #A23B72;">文件操作</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+O</b></td><td style="padding: 5px; border: 1px solid #ddd;">模型反标注</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+U</b></td><td style="padding: 5px; border: 1px solid #ddd;">打开图片所在的文件夹</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+R</b></td><td style="padding: 5px; border: 1px solid #ddd;">打开标签所在的文件夹</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+S</b></td><td style="padding: 5px; border: 1px solid #ddd;">保存</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+S</b></td><td style="padding: 5px; border: 1px solid #ddd;">另存为</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+W</b></td><td style="padding: 5px; border: 1px solid #ddd;">关闭当前文件</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Q</b></td><td style="padding: 5px; border: 1px solid #ddd;">退出程序</td></tr>
</table>

<h3 style="color: #A23B72;">图片导航</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>A</b></td><td style="padding: 5px; border: 1px solid #ddd;">上一张图片</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>D</b></td><td style="padding: 5px; border: 1px solid #ddd;">下一张图片</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Space</b></td><td style="padding: 5px; border: 1px solid #ddd;">裁剪图片</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>E</b></td><td style="padding: 5px; border: 1px solid #ddd;">删除图片</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Q</b></td><td style="padding: 5px; border: 1px solid #ddd;">恢复上一张删除/分类的图片</td></tr>
</table>

<h3 style="color: #A23B72;">图片分类功能</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>0</b></td><td style="padding: 5px; border: 1px solid #ddd;">将当前图片分类到 data_cleaning/0文件夹</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>1</b></td><td style="padding: 5px; border: 1px solid #ddd;">将当前图片分类到 data_cleaning/1文件夹</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>2</b></td><td style="padding: 5px; border: 1px solid #ddd;">将当前图片分类到 data_cleaning/2文件夹</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>3</b></td><td style="padding: 5px; border: 1px solid #ddd;">将当前图片分类到 data_cleaning/3文件夹</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>4</b></td><td style="padding: 5px; border: 1px solid #ddd;">将当前图片分类到 data_cleaning/4文件夹</td></tr>
</table>

<h3 style="color: #A23B72;">标注操作</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>W</b></td><td style="padding: 5px; border: 1px solid #ddd;">创建矩形框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+J</b></td><td style="padding: 5px; border: 1px solid #ddd;">编辑模式</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+D</b></td><td style="padding: 5px; border: 1px solid #ddd;">复制选中的框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Delete</b></td><td style="padding: 5px; border: 1px solid #ddd;">删除选中的框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+V</b></td><td style="padding: 5px; border: 1px solid #ddd;">直接粘贴上一个标注框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+E</b></td><td style="padding: 5px; border: 1px solid #ddd;">编辑标签</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+L</b></td><td style="padding: 5px; border: 1px solid #ddd;">选择线条颜色</td></tr>
</table>

<h3 style="color: #A23B72;">视图操作</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl++</b></td><td style="padding: 5px; border: 1px solid #ddd;">放大</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+-</b></td><td style="padding: 5px; border: 1px solid #ddd;">缩小</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+=</b></td><td style="padding: 5px; border: 1px solid #ddd;">原始大小</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+F</b></td><td style="padding: 5px; border: 1px solid #ddd;">适应窗口</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+F</b></td><td style="padding: 5px; border: 1px solid #ddd;">适应宽度</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+H</b></td><td style="padding: 5px; border: 1px solid #ddd;">隐藏所有标注框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+A</b></td><td style="padding: 5px; border: 1px solid #ddd;">显示所有标注框</td></tr>
</table>

<h3 style="color: #A23B72;">标注框微调</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>←</b></td><td style="padding: 5px; border: 1px solid #ddd;">向左微调标注框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>→</b></td><td style="padding: 5px; border: 1px solid #ddd;">向右微调标注框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>↑</b></td><td style="padding: 5px; border: 1px solid #ddd;">向上微调标注框</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>↓</b></td><td style="padding: 5px; border: 1px solid #ddd;">向下微调标注框</td></tr>
</table>

<h3 style="color: #A23B72;">亮度调节</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+↑</b></td><td style="padding: 5px; border: 1px solid #ddd;">提亮图片</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+↓</b></td><td style="padding: 5px; border: 1px solid #ddd;">变暗图片</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>S</b></td><td style="padding: 5px; border: 1px solid #ddd;">恢复原始亮度</td></tr>
</table>

<h3 style="color: #A23B72;">格式切换</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Y</b></td><td style="padding: 5px; border: 1px solid #ddd;">切换保存格式 (PascalVOC/YOLO/CreateML)</td></tr>
</table>

<h3 style="color: #A23B72;">高级功能</h3>
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+A</b></td><td style="padding: 5px; border: 1px solid #ddd;">切换高级模式</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+S</b></td><td style="padding: 5px; border: 1px solid #ddd;">单类别模式</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+P</b></td><td style="padding: 5px; border: 1px solid #ddd;">显示/隐藏标签文本</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+R</b></td><td style="padding: 5px; border: 1px solid #ddd;">绘制正方形模式</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Shift+L</b></td><td style="padding: 5px; border: 1px solid #ddd;">显示/隐藏标签列表</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+T</b></td><td style="padding: 5px; border: 1px solid #ddd;">视频固定间隔拆帧</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+U</b></td><td style="padding: 5px; border: 1px solid #ddd;">区间均匀取图</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+Z</b></td><td style="padding: 5px; border: 1px solid #ddd;">目标追踪取图</td></tr>
<tr><td style="padding: 5px; border: 1px solid #ddd;"><b>Ctrl+M</b></td><td style="padding: 5px; border: 1px solid #ddd;">人工精准取图</td></tr>
</table>

<p style="margin-top: 20px; color: #666; font-style: italic;">
提示：按住 Ctrl 键拖拽可以绘制正方形标注框
</p>
        """
        
        # 创建文本标签显示快捷键信息
        info_label = QLabel(shortcuts_info)
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setAlignment(Qt.AlignTop)
        info_label.setStyleSheet("QLabel { padding: 10px; }")
        
        scroll_layout.addWidget(info_label)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        
        layout.addWidget(scroll_area)
        
        # 添加关闭按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(shortcuts_dialog.accept)
        close_button.setMinimumWidth(80)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        shortcuts_dialog.setLayout(layout)
        
        # 显示对话框
        shortcuts_dialog.exec_()

    def create_shape(self):
        assert self.beginner()
        self.canvas.set_editing(False)
        self.actions.create.setEnabled(False)

    def toggle_drawing_sensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.set_editing(True)
            self.canvas.restore_cursor()
            self.actions.create.setEnabled(True)

    def toggle_draw_mode(self, edit=True):
        self.canvas.set_editing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def set_create_mode(self):
        assert self.advanced()
        self.toggle_draw_mode(False)

    def set_edit_mode(self):
        assert self.advanced()
        self.toggle_draw_mode(True)
        self.label_selection_changed()

    def update_file_menu(self):
        curr_file_path = self.file_path

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recent_files if f !=
                 curr_file_path and exists(f)]
        for i, f in enumerate(files):
            icon = new_icon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.load_recent, f))
            menu.addAction(action)

    def pop_label_list_menu(self, point):
        self.menus.labelList.exec_(self.label_list.mapToGlobal(point))

    def edit_label(self):
        if not self.canvas.editing():
            return
        item = self.current_item()
        if not item:
            return
        text = self.label_dialog.pop_up(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generate_color_by_text(text))
            self.set_dirty()
            # self.update_combo_box()  # 注释掉组合框更新

    # Tzutalin 20160906 : Add file list and dock to move faster
    def file_item_double_clicked(self, item=None):
        self.cur_img_idx = self.m_img_list.index(ustr(item.text()))
        filename = self.m_img_list[self.cur_img_idx]
        if filename:
            self.load_file(filename)
            # 保存当前进度
            self.save_progress()

    # Add chris - method removed as diffc_button was removed

    # React to canvas signals.
    def shape_selection_changed(self, selected=False):
        if self._no_selection_slot:
            self._no_selection_slot = False
        else:
            shape = self.canvas.selected_shape
            if shape:
                self.shapes_to_items[shape].setSelected(True)
            else:
                self.label_list.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def add_label(self, shape):
        shape.paint_label = self.display_label_option.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generate_color_by_text(shape.label))
        self.items_to_shapes[item] = shape
        self.shapes_to_items[shape] = item
        self.label_list.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        # self.update_combo_box()  # 注释掉组合框更新

    def remove_label(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapes_to_items[shape]
        self.label_list.takeItem(self.label_list.row(item))
        del self.shapes_to_items[shape]
        del self.items_to_shapes[item]
        # self.update_combo_box()  # 注释掉组合框更新

    def load_labels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snap_point_to_canvas(x, y)
                if snapped:
                    self.set_dirty()

                shape.add_point(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generate_color_by_text(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generate_color_by_text(label)

            self.add_label(shape)
        # self.update_combo_box()  # 注释掉组合框更新
        self.canvas.load_shapes(s)

    # def update_combo_box(self):
    #     # Get the unique labels and add them to the Combobox.
    #     items_text_list = [str(self.label_list.item(i).text()) for i in range(self.label_list.count())]
    # 
    #     unique_text_list = list(set(items_text_list))
    #     # Add a null row for showing all the labels
    #     unique_text_list.append("")
    #     unique_text_list.sort()
    # 
    #     self.combo_box.update_items(unique_text_list)

    def save_labels(self, annotation_file_path):
        annotation_file_path = ustr(annotation_file_path)
        if self.label_file is None:
            self.label_file = LabelFile()
            self.label_file.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add different annotation formats here
        try:
            if self.label_file_format == LabelFileFormat.PASCAL_VOC:
                if annotation_file_path[-4:].lower() != ".xml":
                    annotation_file_path += XML_EXT
                self.label_file.save_pascal_voc_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                       self.line_color.getRgb(), self.fill_color.getRgb())
            elif self.label_file_format == LabelFileFormat.YOLO:
                if annotation_file_path[-4:].lower() != ".txt":
                    annotation_file_path += TXT_EXT
                self.label_file.save_yolo_format(annotation_file_path, shapes, self.file_path, self.image_data, self.label_hist,
                                                 self.line_color.getRgb(), self.fill_color.getRgb())
            elif self.label_file_format == LabelFileFormat.CREATE_ML:
                if annotation_file_path[-5:].lower() != ".json":
                    annotation_file_path += JSON_EXT
                self.label_file.save_create_ml_format(annotation_file_path, shapes, self.file_path, self.image_data,
                                                      self.label_hist, self.line_color.getRgb(), self.fill_color.getRgb())
            else:
                self.label_file.save(annotation_file_path, shapes, self.file_path, self.image_data,
                                     self.line_color.getRgb(), self.fill_color.getRgb())
            print('Image:{0} -> Annotation:{1}'.format(self.file_path, annotation_file_path))
            return True
        except LabelFileError as e:
            self.error_message(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copy_selected_shape(self):
        self.add_label(self.canvas.copy_selected_shape())
        # fix copy and delete
        self.shape_selection_changed(True)

    # def combo_selection_changed(self, index):
    #     text = self.combo_box.cb.itemText(index)
    #     for i in range(self.label_list.count()):
    #         if text == "":
    #             self.label_list.item(i).setCheckState(2)
    #         elif text != self.label_list.item(i).text():
    #             self.label_list.item(i).setCheckState(0)
    #         else:
    #             self.label_list.item(i).setCheckState(2)

    # Method removed as default_label_combo_box was removed

    def label_selection_changed(self):
        item = self.current_item()
        if item and self.canvas.editing():
            self._no_selection_slot = True
            self.canvas.select_shape(self.items_to_shapes[item])
            shape = self.items_to_shapes[item]
            # diffc_button reference removed

    def label_item_changed(self, item):
        shape = self.items_to_shapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generate_color_by_text(shape.label)
            self.set_dirty()
        else:  # User probably changed item visibility
            self.canvas.set_shape_visible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if len(self.label_hist) > 0:
            self.label_dialog = LabelDialog(parent=self, list_item=self.label_hist)

        # Sync single class mode from PR#106
        if self.single_class_mode.isChecked() and self.lastLabel:
            text = self.lastLabel
        else:
            text = self.label_dialog.pop_up(text=self.prev_label_text)
            self.lastLabel = text

        if text is not None:
            self.prev_label_text = text
            generate_color = generate_color_by_text(text)
            shape = self.canvas.set_last_label(text, generate_color, generate_color)
            self.add_label(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.set_editing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.set_dirty()

            if text not in self.label_hist:
                self.label_hist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.reset_all_lines()

    def scroll_request(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scroll_bars[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))

    def set_zoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoom_mode = self.MANUAL_ZOOM
        # Arithmetic on scaling factor often results in float
        # Convert to int to avoid type errors
        self.zoom_widget.setValue(int(value))

    def add_zoom(self, increment=10):
        self.set_zoom(self.zoom_widget.value() + increment)

    def zoom_request(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scroll_bars[Qt.Horizontal]
        v_bar = self.scroll_bars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scroll_area.width()
        h = self.scroll_area.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta // (8 * 15)
        scale = 10
        self.add_zoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = int(h_bar.value() + move_x * d_h_bar_max)
        new_v_bar_value = int(v_bar.value() + move_y * d_v_bar_max)

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def light_request(self, delta):
        self.add_light(5*delta // (8 * 15))

    def set_fit_window(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def set_fit_width(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoom_mode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def set_light(self, value):
        self.actions.lightOrg.setChecked(int(value) == 50)
        # Arithmetic on scaling factor often results in float
        # Convert to int to avoid type errors
        self.light_widget.setValue(int(value))

    def add_light(self, increment=10):
        self.set_light(self.light_widget.value() + increment)

    def toggle_polygons(self, value):
        for item, shape in self.items_to_shapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def load_file(self, file_path=None, auto_load_annotations=True):
        """Load the specified file, or the last opened file if None."""
        self.reset_state()
        self.canvas.setEnabled(False)
        if file_path is None:
            file_path = self.settings.get(SETTING_FILENAME)
        # Make sure that filePath is a regular python string, rather than QString
        file_path = ustr(file_path)

        # Fix bug: An  index error after select a directory when open a new file.
        unicode_file_path = ustr(file_path)
        unicode_file_path = os.path.abspath(unicode_file_path)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicode_file_path and self.file_list_widget.count() > 0:
            if unicode_file_path in self.m_img_list:
                index = self.m_img_list.index(unicode_file_path)
                file_widget_item = self.file_list_widget.item(index)
                file_widget_item.setSelected(True)
            else:
                self.file_list_widget.clear()
                self.m_img_list.clear()

        if unicode_file_path and os.path.exists(unicode_file_path):
            if LabelFile.is_label_file(unicode_file_path):
                try:
                    self.label_file = LabelFile(unicode_file_path)
                except LabelFileError as e:
                    self.error_message(u'Error opening file',
                                       (u"<p><b>%s</b></p>"
                                        u"<p>Make sure <i>%s</i> is a valid label file.")
                                       % (e, unicode_file_path))
                    self.status("Error reading %s" % unicode_file_path)
                    
                    return False
                self.image_data = self.label_file.image_data
                self.line_color = QColor(*self.label_file.lineColor)
                self.fill_color = QColor(*self.label_file.fillColor)
                self.canvas.verified = self.label_file.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.image_data = read(unicode_file_path, None)
                self.label_file = None
                self.canvas.verified = False

            if isinstance(self.image_data, QImage):
                image = self.image_data
            else:
                image = QImage.fromData(self.image_data)
            if image.isNull():
                self.error_message(u'Error opening file',
                                   u"<p>Make sure <i>%s</i> is a valid image file." % unicode_file_path)
                self.status("Error reading %s" % unicode_file_path)
                return False
            self.status("Loaded %s" % os.path.basename(unicode_file_path))
            self.image = image
            self.file_path = unicode_file_path
            self.canvas.load_pixmap(QPixmap.fromImage(image))
            if self.label_file:
                self.load_labels(self.label_file.shapes)
            self.set_clean()
            self.canvas.setEnabled(True)
            self.adjust_scale(initial=True)
            self.paint_canvas()
            self.add_recent_file(self.file_path)
            self.toggle_actions(True)
            if auto_load_annotations:
                self.show_bounding_box_from_annotation_file(self.file_path)

            counter = self.counter_str()
            self.setWindowTitle(__appname__ + ' ' + file_path + ' ' + counter)

            # Default : select last item if there is at least one item
            if self.label_list.count():
                self.label_list.setCurrentItem(self.label_list.item(self.label_list.count() - 1))
                self.label_list.item(self.label_list.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def counter_str(self):
        """
        Converts image counter to string representation.
        """
        return '[{} / {}]'.format(self.cur_img_idx + 1, self.img_count)

    def show_bounding_box_from_annotation_file(self, file_path):
        # 检查 file_path 是否为 None，如果是则直接返回
        if file_path is None:
            return
            
        if self.default_save_dir is not None:
            basename = os.path.basename(os.path.splitext(file_path)[0])
            xml_path = os.path.join(self.default_save_dir, basename + XML_EXT)
            txt_path = os.path.join(self.default_save_dir, basename + TXT_EXT)
            json_path = os.path.join(self.default_save_dir, basename + JSON_EXT)

            """Annotation file priority:
            PascalXML > YOLO
            """
            if os.path.isfile(xml_path):
                self.load_pascal_xml_by_filename(xml_path)
            elif os.path.isfile(txt_path):
                self.load_yolo_txt_by_filename(txt_path)
            elif os.path.isfile(json_path):
                self.load_create_ml_json_by_filename(json_path, file_path)

        else:
            xml_path = os.path.splitext(file_path)[0] + XML_EXT
            txt_path = os.path.splitext(file_path)[0] + TXT_EXT
            json_path = os.path.splitext(file_path)[0] + JSON_EXT

            if os.path.isfile(xml_path):
                self.load_pascal_xml_by_filename(xml_path)
            elif os.path.isfile(txt_path):
                self.load_yolo_txt_by_filename(txt_path)
            elif os.path.isfile(json_path):
                self.load_create_ml_json_by_filename(json_path, file_path)
            

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoom_mode != self.MANUAL_ZOOM:
            self.adjust_scale()
        super(MainWindow, self).resizeEvent(event)

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.overlay_color = self.light_widget.color()
        self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        self.zoom_widget.setValue(int(100 * value))

    def scale_fit_window(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scale_fit_width(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.may_continue():
            event.ignore()
            return
        
        # 检查是否存在progress.txt文件，询问用户是否保留
        if self.current_project_dir:
            progress_file_path = os.path.join(self.current_project_dir, 'progress.txt')
            if os.path.exists(progress_file_path):
                # 创建询问对话框
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("保留进度文件")
                msg_box.setText("检测到标注进度文件 progress.txt\n\n是否要保留此文件？")
                msg_box.setInformativeText("保留：下次打开时可以恢复标注进度\n删除：清理临时文件，从头开始标注")
                
                # 添加自定义按钮
                keep_button = msg_box.addButton("保留文件", QMessageBox.YesRole)
                delete_button = msg_box.addButton("删除文件", QMessageBox.NoRole)
                cancel_button = msg_box.addButton("取消退出", QMessageBox.RejectRole)
                
                msg_box.setDefaultButton(keep_button)
                msg_box.exec_()
                
                clicked_button = msg_box.clickedButton()
                
                if clicked_button == cancel_button:
                    # 用户取消退出
                    event.ignore()
                    return
                elif clicked_button == delete_button:
                    # 用户选择删除progress.txt文件
                    try:
                        os.remove(progress_file_path)
                        print("已删除progress.txt文件")
                    except Exception as e:
                        print(f"删除progress.txt文件时出错: {e}")
                # 如果用户选择保留文件，则不做任何操作
        
        # 在程序退出时保存进度（如果文件没有被删除）
        if self.current_project_dir:
            progress_file_path = os.path.join(self.current_project_dir, 'progress.txt')
            if not os.path.exists(progress_file_path):
                # 如果文件被删除了，就不保存进度了
                pass
            else:
                # 文件存在，保存当前进度
                self.save_progress()
        
        settings = self.settings
        # If it loads images from dir, don't load it at the beginning
        if self.dir_name is None:
            settings[SETTING_FILENAME] = self.file_path if self.file_path else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.line_color
        settings[SETTING_FILL_COLOR] = self.fill_color
        settings[SETTING_RECENT_FILES] = self.recent_files
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.default_save_dir and os.path.exists(self.default_save_dir):
            settings[SETTING_SAVE_DIR] = ustr(self.default_save_dir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.last_open_dir and os.path.exists(self.last_open_dir):
            settings[SETTING_LAST_OPEN_DIR] = self.last_open_dir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.auto_saving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.single_class_mode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.display_label_option.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.draw_squares_option.isChecked()
        settings[SETTING_LABEL_FILE_FORMAT] = self.label_file_format
        settings.save()

    def load_recent(self, filename):
        if self.may_continue():
            self.load_file(filename)

    def scan_all_images(self, folder_path):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relative_path = os.path.join(root, file)
                    path = ustr(os.path.abspath(relative_path))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def change_save_dir_dialog(self, _value=False):
        if self.default_save_dir is not None:
            path = ustr(self.default_save_dir)
        else:
            path = '.'

        dir_path = ustr(QFileDialog.getExistingDirectory(self,
                                                         '%s - 选择标签文件夹' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                         | QFileDialog.DontResolveSymlinks))

        if dir_path is not None and len(dir_path) > 1:
            self.default_save_dir = dir_path

        self.show_bounding_box_from_annotation_file(self.file_path)

        self.statusBar().showMessage('%s . 标签将保存到 %s' %
                                     ('已更改保存文件夹', self.default_save_dir))
        self.statusBar().show()


    def open_annotation_dialog(self, _value=False):
        if self.file_path is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.file_path))\
            if self.file_path else '.'
        if self.label_file_format == LabelFileFormat.PASCAL_VOC:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self, '%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.load_pascal_xml_by_filename(filename)

        elif self.label_file_format == LabelFileFormat.YOLO:
            # 处理YOLO格式标签文件
            filters = "Open Annotation TXT file (%s)" % ' '.join(['*.txt'])
            filename = ustr(QFileDialog.getOpenFileName(self, '%s - Choose a txt file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
                
                # 检查classes.txt文件是否存在
                label_dir = os.path.dirname(filename)
                classes_file = os.path.join(label_dir, "classes.txt")
                
                if not os.path.exists(classes_file):
                    # 创建空的classes.txt文件
                    try:
                        with open(classes_file, 'w', encoding='utf-8') as f:
                            f.write("")  # 创建空文件
                        
                        # 导入类别定义对话框
                        from libs.classDefinitionDialog import ClassDefinitionDialog
                        
                        # 显示类别定义对话框
                        dialog = ClassDefinitionDialog(classes_file, self)
                        if dialog.exec_() == QDialog.Accepted:
                            # 用户确认了类别定义，加载YOLO标签文件
                            self.load_yolo_txt_by_filename(filename)
                        else:
                            # 用户取消了，删除空的classes.txt文件
                            if os.path.exists(classes_file):
                                os.remove(classes_file)
                    except Exception as e:
                        QMessageBox.critical(self, "错误", f"创建classes.txt文件失败:\n{str(e)}")
                else:
                    # classes.txt文件存在，直接加载
                    self.load_yolo_txt_by_filename(filename)

        elif self.label_file_format == LabelFileFormat.CREATE_ML:
            
            filters = "Open Annotation JSON file (%s)" % ' '.join(['*.json'])
            filename = ustr(QFileDialog.getOpenFileName(self, '%s - Choose a json file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]

            self.load_create_ml_json_by_filename(filename, self.file_path)         
        

    def open_dir_dialog(self, _value=False, dir_path=None, silent=False):
        if not self.may_continue():
            return

        default_open_dir_path = dir_path if dir_path else '.'
        if self.last_open_dir and os.path.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = os.path.dirname(self.file_path) if self.file_path else '.'
        if silent != True:
            target_dir_path = ustr(QFileDialog.getExistingDirectory(self,
                                                                    '%s - 选择图片文件夹' % __appname__, default_open_dir_path,
                                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            target_dir_path = ustr(default_open_dir_path)
        self.last_open_dir = target_dir_path
        self.import_dir_images(target_dir_path)
        self.default_save_dir = target_dir_path
        if self.file_path:
            self.show_bounding_box_from_annotation_file(file_path=self.file_path)

    def import_dir_images(self, dir_path):
        if not self.may_continue() or not dir_path:
            return

        self.last_open_dir = dir_path
        self.dir_name = dir_path
        self.current_project_dir = dir_path  # 设置当前项目目录
        self.file_path = None
        self.file_list_widget.clear()
        self.m_img_list = self.scan_all_images(dir_path)
        self.img_count = len(self.m_img_list)
        
        # 检查是否有保存的进度，但在撤回操作时跳过恢复对话框
        progress_data = self.load_progress()
        if progress_data and progress_data['current_index'] > 0 and progress_data['current_index'] < self.img_count:
            # 如果正在进行撤回操作，直接恢复进度而不显示对话框
            if self.is_restoring_operation:
                self.restore_progress_without_auto_save(progress_data['current_index'])
            elif self.ask_restore_progress(progress_data):
                # 恢复进度时，先加载图片但不自动触发标签目录选择
                self.restore_progress_without_auto_save(progress_data['current_index'])
            else:
                self.open_next_image_without_auto_save()
        else:
            self.open_next_image_without_auto_save()
            
        for imgPath in self.m_img_list:
            item = QListWidgetItem(imgPath)
            self.file_list_widget.addItem(item)

    def verify_image(self, _value=False):
        # 打开裁剪对话框
        if self.file_path is not None:
            # 传递当前图片路径和标签文件夹路径
            label_dir = self.default_save_dir if self.default_save_dir else os.path.dirname(self.file_path)
            crop_dialog = CropDialog(current_image_path=self.file_path, label_dir=label_dir, parent=self)
            crop_dialog.exec_()

    def open_prev_image(self, _value=False):
        # Proceeding prev image without dialog if having any label
        if self.auto_saving.isChecked():
            if self.default_save_dir is not None:
                if self.dirty is True:
                    self.save_file()
            else:
                self.change_save_dir_dialog()
                return

        if not self.may_continue():
            return

        if self.img_count <= 0:
            return

        if self.file_path is None:
            return

        if self.cur_img_idx - 1 >= 0:
            self.cur_img_idx -= 1
            filename = self.m_img_list[self.cur_img_idx]
            if filename:
                self.load_file(filename)
                # 保存当前进度
                self.save_progress()

    def open_next_image(self, _value=False):
        # Proceeding next image without dialog if having any label
        if self.auto_saving.isChecked():
            if self.default_save_dir is not None:
                if self.dirty is True:
                    self.save_file()
            else:
                self.change_save_dir_dialog()
                return

        if not self.may_continue():
            return

        if self.img_count <= 0:
            return
        
        if not self.m_img_list:
            return

        filename = None
        if self.file_path is None:
            filename = self.m_img_list[0]
            self.cur_img_idx = 0
        else:
            if self.cur_img_idx + 1 < self.img_count:
                self.cur_img_idx += 1
                filename = self.m_img_list[self.cur_img_idx]

        if filename:
            self.load_file(filename)
            # 保存当前进度
            self.save_progress()
            
            # 检查是否已经查看到最后一张图片
            if self.cur_img_idx == self.img_count - 1:
                # 自动删除progress.txt文件
                self.auto_delete_progress_file()

    def open_file(self, _value=False):
        """打开模型反标注选择对话框"""
        # 创建模型选择对话框
        model_choice_dialog = QDialog(self)
        model_choice_dialog.setWindowTitle("选择模型类型")
        model_choice_dialog.setFixedSize(500, 300)  # 增大对话框尺寸，提高可读性
        model_choice_dialog.setModal(True)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 添加说明标签
        label = QLabel("请选择要使用的模型类型：")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # YOLO模型按钮
        yolo_button = QPushButton("YOLO模型")
        yolo_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # NanoDet模型按钮
        nanodet_button = QPushButton("NanoDet模型")
        nanodet_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        # 添加按钮到布局
        button_layout.addWidget(yolo_button)
        button_layout.addWidget(nanodet_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        model_choice_dialog.setLayout(layout)
        
        # 按钮事件处理
        def open_yolo_dialog():
            model_choice_dialog.accept()
            dialog = ModelInferenceDialog(self)
            dialog.exec_()
            
        def open_nanodet_dialog():
            model_choice_dialog.accept()
            dialog = NanoDetInferenceDialog(self)
            dialog.exec_()
            
        def cancel_dialog():
            model_choice_dialog.reject()
        
        # 连接按钮信号
        yolo_button.clicked.connect(open_yolo_dialog)
        nanodet_button.clicked.connect(open_nanodet_dialog)
        cancel_button.clicked.connect(cancel_dialog)
        
        # 显示对话框
        model_choice_dialog.exec_()

    def save_file(self, _value=False):
        if self.default_save_dir is not None and len(ustr(self.default_save_dir)):
            if self.file_path:
                image_file_name = os.path.basename(self.file_path)
                saved_file_name = os.path.splitext(image_file_name)[0]
                saved_path = os.path.join(ustr(self.default_save_dir), saved_file_name)
                self._save_file(saved_path)
        else:
            image_file_dir = os.path.dirname(self.file_path)
            image_file_name = os.path.basename(self.file_path)
            saved_file_name = os.path.splitext(image_file_name)[0]
            saved_path = os.path.join(image_file_dir, saved_file_name)
            self._save_file(saved_path if self.label_file
                            else self.save_file_dialog(remove_ext=False))

    def save_file_as(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._save_file(self.save_file_dialog())

    def save_file_dialog(self, remove_ext=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        open_dialog_path = self.current_path()
        dlg = QFileDialog(self, caption, open_dialog_path, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filename_without_extension = os.path.splitext(self.file_path)[0]
        dlg.selectFile(filename_without_extension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            full_file_path = ustr(dlg.selectedFiles()[0])
            if remove_ext:
                return os.path.splitext(full_file_path)[0]  # Return file path without the extension.
            else:
                return full_file_path
        return ''

    def _save_file(self, annotation_file_path):
        if annotation_file_path and self.save_labels(annotation_file_path):
            self.set_clean()
            self.statusBar().showMessage('Saved to  %s' % annotation_file_path)
            self.statusBar().show()

    def close_file(self, _value=False):
        if not self.may_continue():
            return
        self.reset_state()
        self.set_clean()
        self.toggle_actions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def delete_image(self):
        """删除当前图片和对应标签 - 移动到data_cleaning文件夹而不是永久删除"""
        delete_path = self.file_path
        if delete_path is not None:
            idx = self.cur_img_idx
            if os.path.exists(delete_path):
                # 获取当前图片所在目录
                current_dir = os.path.dirname(delete_path)
                # 创建data_cleaning文件夹路径（与当前目录同级）
                parent_dir = os.path.dirname(current_dir)
                data_cleaning_folder = os.path.join(parent_dir, 'data_cleaning')
                delete_images_folder = os.path.join(data_cleaning_folder, 'Delete_images')
                delete_labels_folder = os.path.join(data_cleaning_folder, 'Delete_labels')
                
                # 如果文件夹不存在，创建它们
                if not os.path.exists(delete_images_folder):
                    os.makedirs(delete_images_folder)
                if not os.path.exists(delete_labels_folder):
                    os.makedirs(delete_labels_folder)
                
                # 获取文件名
                filename = os.path.basename(delete_path)
                # 图片目标路径
                image_target_path = os.path.join(delete_images_folder, filename)
                
                # 如果目标文件已存在，添加数字后缀
                counter = 1
                original_image_target = image_target_path
                while os.path.exists(image_target_path):
                    name, ext = os.path.splitext(original_image_target)
                    image_target_path = f"{name}_{counter}{ext}"
                    counter += 1
                
                try:
                    # 移动图片文件到Delete_images文件夹
                    import shutil
                    shutil.move(delete_path, image_target_path)
                    
                    # 查找并移动对应的标注文件
                    moved_labels = []  # 记录移动的标签文件信息
                    annotation_extensions = ['.xml', '.txt', '.json']
                    
                    # 首先在图片同目录查找标签文件
                    image_dir = os.path.dirname(delete_path)
                    image_name = os.path.splitext(os.path.basename(delete_path))[0]
                    
                    for ext in annotation_extensions:
                        annotation_path = os.path.join(image_dir, image_name + ext)
                        if os.path.exists(annotation_path):
                            annotation_filename = os.path.basename(annotation_path)
                            annotation_target = os.path.join(delete_labels_folder, annotation_filename)
                            
                            # 处理标注文件重名
                            counter = 1
                            original_annotation_target = annotation_target
                            while os.path.exists(annotation_target):
                                name, ext_part = os.path.splitext(original_annotation_target)
                                annotation_target = f"{name}_{counter}{ext_part}"
                                counter += 1
                            
                            shutil.move(annotation_path, annotation_target)
                            moved_labels.append((annotation_path, annotation_target))
                            print(f"标签文件已移动到: {annotation_target}")
                    
                    # 如果设置了default_save_dir，也在该目录查找标签文件
                    if self.default_save_dir and self.default_save_dir != image_dir:
                        for ext in annotation_extensions:
                            annotation_path = os.path.join(self.default_save_dir, image_name + ext)
                            if os.path.exists(annotation_path):
                                annotation_filename = os.path.basename(annotation_path)
                                annotation_target = os.path.join(delete_labels_folder, annotation_filename)
                                
                                # 处理标注文件重名
                                counter = 1
                                original_annotation_target = annotation_target
                                while os.path.exists(annotation_target):
                                    name, ext_part = os.path.splitext(original_annotation_target)
                                    annotation_target = f"{name}_{counter}{ext_part}"
                                    counter += 1
                                
                                shutil.move(annotation_path, annotation_target)
                                moved_labels.append((annotation_path, annotation_target))
                                print(f"标签文件已移动到: {annotation_target}")
                    
                    # 记录删除操作，用于撤回功能（包含图片和所有标签文件的信息）
                    self.deleted_images.append({
                        'image': (delete_path, image_target_path),
                        'labels': moved_labels,
                        'original_index': idx  # 记录原始位置索引
                    })
                    
                    # 添加到操作历史记录
                    self.operation_history.append({
                        'type': 'delete',
                        'timestamp': time.time()
                    })
                    
                    print(f"图片已移动到: {image_target_path}")
                    if moved_labels:
                        print(f"同时移动了 {len(moved_labels)} 个标签文件")
                    
                except Exception as e:
                    print(f"移动文件时出错: {e}")
                    return
            
            # 直接更新图片列表，避免触发进度恢复对话框
            # 从当前图片列表中移除已删除的图片
            if delete_path in self.m_img_list:
                self.m_img_list.remove(delete_path)
                self.img_count = len(self.m_img_list)
                
                # 更新文件列表控件
                self.file_list_widget.clear()
                for imgPath in self.m_img_list:
                    item = QListWidgetItem(imgPath)
                    self.file_list_widget.addItem(item)
                
                # 调整当前图片索引并加载下一张图片
                if self.img_count > 0:
                    # 如果删除的是最后一张图片，索引需要减1
                    if idx >= self.img_count:
                        self.cur_img_idx = self.img_count - 1
                    else:
                        self.cur_img_idx = idx
                    
                    # 加载当前索引对应的图片
                    filename = self.m_img_list[self.cur_img_idx]
                    self.load_file(filename)
                    
                    # 保存当前进度（不会触发进度恢复对话框）
                    self.save_progress()
                else:
                    # 如果没有图片了，关闭文件
                    self.close_file()

    def restore_image(self):
        """恢复上一张删除的图片和对应标签，并回到删除时的位置"""
        if not self.deleted_images:
            print("没有可恢复的图片")
            return
        
        # 获取最后一次删除的图片信息（新的数据结构）
        delete_info = self.deleted_images.pop()
        original_image_path, deleted_image_path = delete_info['image']
        moved_labels = delete_info['labels']
        original_index = delete_info['original_index']
        
        if os.path.exists(deleted_image_path):
            try:
                import shutil
                # 恢复图片文件
                shutil.move(deleted_image_path, original_image_path)
                print(f"图片已恢复到: {original_image_path}")
                
                # 恢复所有对应的标注文件
                restored_labels_count = 0
                for original_label_path, deleted_label_path in moved_labels:
                    if os.path.exists(deleted_label_path):
                        shutil.move(deleted_label_path, original_label_path)
                        restored_labels_count += 1
                        print(f"标签文件已恢复到: {original_label_path}")
                
                if restored_labels_count > 0:
                    print(f"共恢复了 {restored_labels_count} 个标签文件")
                
                # 智能恢复图片列表和位置，确保丝滑的用户体验
                self._smart_restore_image_list_and_position(original_image_path, original_index)
                
            except Exception as e:
                print(f"恢复文件时出错: {e}")
                # 如果恢复失败，重新添加到删除列表
                self.deleted_images.append(delete_info)
        else:
            print(f"删除的图片文件不存在: {deleted_image_path}")
            # 文件不存在，从删除列表中移除这个记录

    def classify_image(self, category):
        """将当前图片分类到指定类别文件夹，图片到data_cleaning/category/images/，标签到data_cleaning/category/labels/"""
        if self.file_path is None:
            print("没有当前图片可以分类")
            return
        
        current_image_path = self.file_path
        if not os.path.exists(current_image_path):
            print("当前图片文件不存在")
            return
        
        # 获取当前图片所在目录
        current_dir = os.path.dirname(current_image_path)
        # 获取图片文件夹的父目录，在父目录下创建data_cleaning文件夹
        parent_dir = os.path.dirname(current_dir)
        data_cleaning_dir = os.path.join(parent_dir, 'data_cleaning')
        category_folder = os.path.join(data_cleaning_dir, str(category))
        
        # 创建images和labels子文件夹
        images_folder = os.path.join(category_folder, 'images')
        labels_folder = os.path.join(category_folder, 'labels')
        
        # 如果文件夹不存在，创建它们
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
            print(f"创建图片分类文件夹: {images_folder}")
        
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)
            print(f"创建标签分类文件夹: {labels_folder}")
        
        # 获取文件名
        filename = os.path.basename(current_image_path)
        # 图片目标路径（放到images文件夹）
        image_target_path = os.path.join(images_folder, filename)
        
        # 如果目标文件已存在，添加数字后缀
        counter = 1
        original_image_target = image_target_path
        while os.path.exists(image_target_path):
            name, ext = os.path.splitext(original_image_target)
            image_target_path = f"{name}_{counter}{ext}"
            counter += 1
        
        try:
            import shutil
            # 移动图片文件到images文件夹
            shutil.move(current_image_path, image_target_path)
            
            # 查找并移动对应的标注文件到labels文件夹
            moved_labels = []  # 记录移动的标签文件信息
            annotation_extensions = ['.xml', '.txt', '.json']
            
            # 首先在图片同目录查找标签文件
            image_dir = os.path.dirname(current_image_path)
            image_name = os.path.splitext(os.path.basename(current_image_path))[0]
            
            for ext in annotation_extensions:
                annotation_path = os.path.join(image_dir, image_name + ext)
                if os.path.exists(annotation_path):
                    annotation_filename = os.path.basename(annotation_path)
                    # 标签文件放到labels文件夹
                    annotation_target = os.path.join(labels_folder, annotation_filename)
                    
                    # 处理标注文件重名
                    counter = 1
                    original_annotation_target = annotation_target
                    while os.path.exists(annotation_target):
                        name, ext_part = os.path.splitext(original_annotation_target)
                        annotation_target = f"{name}_{counter}{ext_part}"
                        counter += 1
                    
                    shutil.move(annotation_path, annotation_target)
                    moved_labels.append((annotation_path, annotation_target))
                    print(f"标签文件已移动到: {annotation_target}")
            
            # 如果设置了default_save_dir，也在该目录查找标签文件
            if self.default_save_dir and self.default_save_dir != image_dir:
                for ext in annotation_extensions:
                    annotation_path = os.path.join(self.default_save_dir, image_name + ext)
                    if os.path.exists(annotation_path):
                        annotation_filename = os.path.basename(annotation_path)
                        # 标签文件放到labels文件夹
                        annotation_target = os.path.join(labels_folder, annotation_filename)
                        
                        # 处理标注文件重名
                        counter = 1
                        original_annotation_target = annotation_target
                        while os.path.exists(annotation_target):
                            name, ext_part = os.path.splitext(original_annotation_target)
                            annotation_target = f"{name}_{counter}{ext_part}"
                            counter += 1
                        
                        shutil.move(annotation_path, annotation_target)
                        moved_labels.append((annotation_path, annotation_target))
                        print(f"标签文件已移动到: {annotation_target}")
            
            # 记录分类操作，用于撤回功能
            classify_info = {
                'image': (current_image_path, image_target_path),
                'labels': moved_labels,
                'original_index': self.cur_img_idx,  # 记录原始位置索引
                'category': category
            }
            self.classified_images.append(classify_info)
            
            # 添加到操作历史记录
            self.operation_history.append({
                'type': 'classify',
                'timestamp': time.time()
            })
            
            print(f"图片已分类到类别 {category}/images/: {image_target_path}")
            if moved_labels:
                print(f"同时移动了 {len(moved_labels)} 个标签文件到 {category}/labels/")
            
            # 更新图片列表，移除已分类的图片
            if current_image_path in self.m_img_list:
                self.m_img_list.remove(current_image_path)
                self.img_count = len(self.m_img_list)
                
                # 更新文件列表控件
                self.file_list_widget.clear()
                for imgPath in self.m_img_list:
                    item = QListWidgetItem(imgPath)
                    self.file_list_widget.addItem(item)
                
                # 调整当前图片索引并加载下一张图片
                if self.img_count > 0:
                    # 如果分类的是最后一张图片，索引需要减1
                    if self.cur_img_idx >= self.img_count:
                        self.cur_img_idx = self.img_count - 1
                    
                    # 加载当前索引对应的图片
                    filename = self.m_img_list[self.cur_img_idx]
                    self.load_file(filename)
                    
                    # 保存当前进度
                    self.save_progress()
                else:
                    # 如果没有图片了，关闭文件
                    self.close_file()
            
        except Exception as e:
            print(f"分类图片时出错: {e}")

    def restore_classified_image(self):
        """恢复上一张分类的图片和对应标签，并回到分类时的位置"""
        if not self.classified_images:
            print("没有可恢复的分类图片")
            return
        
        # 获取最后一次分类的图片信息
        classify_info = self.classified_images.pop()
        original_image_path, classified_image_path = classify_info['image']
        moved_labels = classify_info['labels']
        original_index = classify_info['original_index']
        category = classify_info['category']
        
        if os.path.exists(classified_image_path):
            try:
                import shutil
                # 恢复图片文件
                shutil.move(classified_image_path, original_image_path)
                print(f"图片已从类别 {category} 恢复到: {original_image_path}")
                
                # 恢复所有对应的标注文件
                restored_labels_count = 0
                for original_label_path, classified_label_path in moved_labels:
                    if os.path.exists(classified_label_path):
                        shutil.move(classified_label_path, original_label_path)
                        restored_labels_count += 1
                        print(f"标签文件已恢复到: {original_label_path}")
                
                if restored_labels_count > 0:
                    print(f"共恢复了 {restored_labels_count} 个标签文件")
                
                # 使用智能恢复方法来更新图片列表和位置
                self._smart_restore_image_list_and_position(original_image_path, original_index)
                
            except Exception as e:
                print(f"恢复分类图片时出错: {e}")
                # 如果恢复失败，重新添加到分类列表
                self.classified_images.append(classify_info)
        else:
            print(f"分类的图片文件不存在: {classified_image_path}")
            # 文件不存在，从分类列表中移除这个记录

    def restore_last_operation(self):
        """统一的撤回功能，能够撤回删除或分类操作，确保丝滑的用户体验"""
        
        # 检查是否有操作可以撤回
        if not self.operation_history:
            print("没有可撤回的操作")
            return False
        
        # 获取最后一次操作，但先不移除，等确认可以撤回后再移除
        last_operation = self.operation_history[-1]
        operation_type = last_operation['type']
        
        success = False
        
        if operation_type == 'delete':
            # 撤回删除操作
            if self.deleted_images:
                # 先移除操作历史记录
                self.operation_history.pop()
                # 执行撤回
                self.restore_image()
                print("✓ 已撤回删除操作")
                success = True
            else:
                print("⚠ 没有可恢复的删除图片，清理无效的操作历史")
                # 清理无效的操作历史记录
                self.operation_history.pop()
                
        elif operation_type == 'classify':
            # 撤回分类操作
            if self.classified_images:
                # 先移除操作历史记录
                self.operation_history.pop()
                # 执行撤回
                self.restore_classified_image()
                print("✓ 已撤回分类操作")
                success = True
            else:
                print("⚠ 没有可恢复的分类图片，清理无效的操作历史")
                # 清理无效的操作历史记录
                self.operation_history.pop()
                
        else:
            print(f"⚠ 未知的操作类型: {operation_type}，清理无效记录")
            # 清理无效的操作历史记录
            self.operation_history.pop()
        
        return success

    def reset_all(self):
        self.settings.reset()
        self.close()
        process = QProcess()
        process.startDetached(os.path.abspath(__file__))

    def may_continue(self):
        if not self.dirty:
            return True
        else:
            discard_changes = self.discard_changes_dialog()
            if discard_changes == QMessageBox.No:
                return True
            elif discard_changes == QMessageBox.Yes:
                self.save_file()
                return True
            else:
                return False

    def discard_changes_dialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        # 详细说明未保存的更改内容
        msg = u'您有未保存的标注更改，是否要保存并继续？\n\n未保存的更改包括：\n• 新建的标注框\n• 修改的标注框位置或大小\n• 更改的标注框标签名称\n• 删除的标注框\n• 复制或移动的标注框\n• 修改的标注框颜色\n\n点击"否"将撤销所有更改。'
        return QMessageBox.warning(self, u'注意', msg, yes | no | cancel)

    def error_message(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def current_path(self):
        return os.path.dirname(self.file_path) if self.file_path else '.'

    def choose_color1(self):
        color = self.color_dialog.getColor(self.line_color, u'Choose line color',
                                           default=DEFAULT_LINE_COLOR)
        if color:
            self.line_color = color
            Shape.line_color = color
            self.canvas.set_drawing_color(color)
            self.canvas.update()
            self.set_dirty()

    def delete_selected_shape(self):
        self.remove_label(self.canvas.delete_selected())
        self.set_dirty()
        if self.no_shapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def choose_shape_line_color(self):
        color = self.color_dialog.getColor(self.line_color, u'Choose Line Color',
                                           default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selected_shape.line_color = color
            self.canvas.update()
            self.set_dirty()

    def choose_shape_fill_color(self):
        color = self.color_dialog.getColor(self.fill_color, u'Choose Fill Color',
                                           default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selected_shape.fill_color = color
            self.canvas.update()
            self.set_dirty()

    def copy_shape(self):
        if self.canvas.selected_shape is None:
            # True if one accidentally touches the left mouse button before releasing
            return
        self.canvas.end_move(copy=True)
        self.add_label(self.canvas.selected_shape)
        self.set_dirty()

    def move_shape(self):
        self.canvas.end_move(copy=False)
        self.set_dirty()

    def load_predefined_classes(self, predef_classes_file):
        if predef_classes_file is not None and os.path.exists(predef_classes_file) is True:
            with codecs.open(predef_classes_file, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.label_hist is None:
                        self.label_hist = [line]
                    else:
                        self.label_hist.append(line)

    def load_pascal_xml_by_filename(self, xml_path):
        if self.file_path is None:
            return
        if os.path.isfile(xml_path) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        t_voc_parse_reader = PascalVocReader(xml_path)
        shapes = t_voc_parse_reader.get_shapes()
        self.load_labels(shapes)
        self.canvas.verified = t_voc_parse_reader.verified

    def load_yolo_txt_by_filename(self, txt_path):
        if self.file_path is None:
            return
        if os.path.isfile(txt_path) is False:
            return

        # 检查classes.txt文件是否存在
        label_dir = os.path.dirname(txt_path)
        classes_file = os.path.join(label_dir, "classes.txt")
        
        if not os.path.exists(classes_file):
            # 创建空的classes.txt文件
            try:
                with open(classes_file, 'w', encoding='utf-8') as f:
                    f.write("")  # 创建空文件
                
                # 导入类别定义对话框
                from libs.classDefinitionDialog import ClassDefinitionDialog
                
                # 显示类别定义对话框
                dialog = ClassDefinitionDialog(classes_file, self)
                if dialog.exec_() == QDialog.Accepted:
                    # 用户确认了类别定义，继续加载YOLO标签文件
                    pass
                else:
                    # 用户取消了，删除空的classes.txt文件并返回
                    if os.path.exists(classes_file):
                        os.remove(classes_file)
                    return
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建classes.txt文件失败:\n{str(e)}")
                return

        self.set_format(FORMAT_YOLO)
        t_yolo_parse_reader = YoloReader(txt_path, self.image)
        shapes = t_yolo_parse_reader.get_shapes()
        print(shapes)
        self.load_labels(shapes)
        self.canvas.verified = t_yolo_parse_reader.verified

    def load_create_ml_json_by_filename(self, json_path, file_path):
        if self.file_path is None:
            return
        if os.path.isfile(json_path) is False:
            return

        self.set_format(FORMAT_CREATEML)

        create_ml_parse_reader = CreateMLReader(json_path, file_path)
        shapes = create_ml_parse_reader.get_shapes()
        self.load_labels(shapes)
        self.canvas.verified = create_ml_parse_reader.verified

    def copy_previous_bounding_boxes(self):
        current_index = self.m_img_list.index(self.file_path)
        if current_index - 1 >= 0:
            prev_file_path = self.m_img_list[current_index - 1]
            self.show_bounding_box_from_annotation_file(prev_file_path)
            self.save_file()

    def toggle_paint_labels_option(self):
        for shape in self.canvas.shapes:
            shape.paint_label = self.display_label_option.isChecked()

    def toggle_draw_square(self):
        self.canvas.set_drawing_shape_to_square(self.draw_squares_option.isChecked())

    def _smart_restore_image_list_and_position(self, restored_image_path, original_index):
        """
        智能恢复图片列表和位置的方法
        确保恢复的图片能够正确回到原来的位置，提供丝滑的用户体验
        
        参数:
        - restored_image_path: 恢复的图片路径
        - original_index: 原始的图片索引位置
        """
        try:
            # 重新扫描图片目录，更新图片列表
            if self.dir_name:
                # 设置标志位，表示正在进行撤回操作，避免触发恢复对话框
                self.is_restoring_operation = True
                
                # 获取当前目录下的所有图片文件
                self.import_dir_images(self.dir_name)
                
                # 重置标志位
                self.is_restoring_operation = False
                
                # 查找恢复的图片在新列表中的位置
                if restored_image_path in self.m_img_list:
                    # 找到恢复图片的新索引
                    new_index = self.m_img_list.index(restored_image_path)
                    
                    # 设置当前图片索引为恢复图片的位置
                    self.cur_img_idx = new_index
                    
                    # 加载恢复的图片
                    self.load_file(restored_image_path)
                    
                    # 更新文件列表控件的选中状态
                    self.file_list_widget.setCurrentRow(new_index)
                    
                    # 保存当前进度
                    self.save_progress()
                    
                    print(f"✓ 图片已恢复到位置 {new_index + 1}/{self.img_count}: {os.path.basename(restored_image_path)}")
                    
                else:
                    # 如果恢复的图片不在当前目录的图片列表中，尝试使用原始索引
                    print(f"⚠ 恢复的图片不在当前目录中，尝试使用原始位置")
                    
                    # 确保索引在有效范围内
                    if 0 <= original_index < self.img_count:
                        self.cur_img_idx = original_index
                        filename = self.m_img_list[self.cur_img_idx]
                        self.load_file(filename)
                        self.file_list_widget.setCurrentRow(self.cur_img_idx)
                        self.save_progress()
                    elif self.img_count > 0:
                        # 如果原始索引超出范围，使用最后一张图片
                        self.cur_img_idx = self.img_count - 1
                        filename = self.m_img_list[self.cur_img_idx]
                        self.load_file(filename)
                        self.file_list_widget.setCurrentRow(self.cur_img_idx)
                        self.save_progress()
                    else:
                        # 如果没有图片了，关闭文件
                        self.close_file()
            else:
                print("⚠ 没有设置图片目录，无法智能恢复位置")
                
        except Exception as e:
            print(f"智能恢复位置时出错: {e}")
            # 确保在异常情况下也重置标志位
            self.is_restoring_operation = False
            
            # 如果智能恢复失败，尝试简单的位置恢复
            try:
                if 0 <= original_index < self.img_count:
                    self.cur_img_idx = original_index
                    filename = self.m_img_list[self.cur_img_idx]
                    self.load_file(filename)
                    self.file_list_widget.setCurrentRow(self.cur_img_idx)
                    self.save_progress()
                elif self.img_count > 0:
                    self.cur_img_idx = 0
                    filename = self.m_img_list[self.cur_img_idx]
                    self.load_file(filename)
                    self.file_list_widget.setCurrentRow(self.cur_img_idx)
                    self.save_progress()
            except Exception as fallback_error:
                print(f"位置恢复失败: {fallback_error}")

    # 视频拆帧功能方法
    def open_video_frame_fixed(self):
        """打开隔固定帧取图功能"""
        try:
            dialog = VideoFrameExtractorDialog(self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开视频拆帧工具失败:\n{str(e)}")

    def open_video_frame_uniform(self):
        """打开区间均匀取图功能"""
        try:
            # 导入视频标签工具
            from libs.videolabeltool import VideoLabelingTool
            
            # 创建并显示视频标签工具窗口
            self.video_label_tool = VideoLabelingTool()
            self.video_label_tool.show()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开视频标签工具失败:\n{str(e)}")

    def open_video_frame_tracking(self):
        """打开目标追踪取图功能"""
        try:
            # 导入配置对话框
            from libs.video_tracking_dialog import VideoTrackingDialog
            
            # 创建配置对话框
            dialog = VideoTrackingDialog(self)
            
            # 显示对话框 - 对话框内部会处理界面切换
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开目标追踪配置失败:\n{str(e)}")

    def open_video_frame_manual(self):
        """打开人工精准取图功能"""
        try:
            # 导入视频帧捕捉工具
            from libs.frame_capture_CUT import VideoPlayer
            
            # 创建并显示视频帧捕捉工具窗口
            self.video_frame_capture_tool = VideoPlayer()
            self.video_frame_capture_tool.show()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开人工精准取图工具失败:\n{str(e)}")

    # 数据清洗功能方法
    def data_cleaning_duplicate(self):
        """智能图片去重功能"""
        try:
            # 导入智能图片去重工具
            from libs.intelligent_image_deduplication import show_intelligent_image_deduplication
            
            # 显示智能图片去重工具对话框
            show_intelligent_image_deduplication(self)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开智能图片去重工具失败:\n{str(e)}\n\n请确保已安装必要的依赖库：\n- torch\n- torchvision\n- scikit-learn\n- opencv-python\n- Pillow")

    def data_cleaning_blur(self):
        """去模糊功能"""
        try:
            # 导入图像去模糊检测工具
            from libs.image_deblur_detection import show_image_deblur_detection_dialog
            
            # 显示图像去模糊检测工具对话框
            show_image_deblur_detection_dialog(self)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图像去模糊检测工具失败:\n{str(e)}\n\n请确保已安装必要的依赖库：\n- opencv-python\n- numpy\n- Pillow")

    def data_cleaning_overexposure(self):
        """去曝光功能"""
        try:
            # 导入图像曝光检测工具
            from libs.image_exposure_detection import show_image_exposure_detection_dialog
            
            # 显示图像曝光检测工具对话框
            show_image_exposure_detection_dialog(self)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图像曝光检测工具失败:\n{str(e)}\n\n请确保已安装必要的依赖库：\n- opencv-python\n- numpy\n- Pillow")

    def change_theme_color(self):
        """更改主题颜色"""
        try:
            # 创建颜色选择对话框
            color = QColorDialog.getColor(Qt.white, self, "选择主题颜色")
            
            if color.isValid():
                # 应用主题颜色到整个应用程序
                self.apply_theme_color(color)
                
                # 保存主题颜色设置到配置文件
                settings = QSettings()
                settings.setValue('theme_color', color.name())
                
                QMessageBox.information(self, "设置成功", f"主题颜色已更改为: {color.name()}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更改主题颜色失败:\n{str(e)}")
    
    def apply_theme_color(self, color):
        """应用主题颜色到整个应用程序的所有组件"""
        try:
            # 计算文本颜色（根据背景颜色的亮度自动选择黑色或白色）
            text_color = Qt.black if color.lightness() > 128 else Qt.white
            
            # 计算按钮颜色（比主题颜色稍微深一点）
            button_color = color.darker(110)
            
            # 计算高亮颜色（比主题颜色稍微亮一点）
            highlight_color = color.lighter(120)
            
            # 计算输入框背景颜色（比主题颜色稍微亮一点）
            base_color = color.lighter(105)
            
            # 创建全局调色板
            palette = QPalette()
            
            # 设置窗口背景色
            palette.setColor(QPalette.Window, color)
            palette.setColor(QPalette.WindowText, text_color)
            
            # 设置按钮颜色
            palette.setColor(QPalette.Button, button_color)
            palette.setColor(QPalette.ButtonText, text_color)
            
            # 设置输入框颜色
            palette.setColor(QPalette.Base, base_color)
            palette.setColor(QPalette.Text, text_color)
            
            # 设置选中项颜色
            palette.setColor(QPalette.Highlight, highlight_color)
            palette.setColor(QPalette.HighlightedText, text_color)
            
            # 设置工具提示颜色
            palette.setColor(QPalette.ToolTipBase, color)
            palette.setColor(QPalette.ToolTipText, text_color)
            
            # 设置替代背景色（用于表格等）
            palette.setColor(QPalette.AlternateBase, color.darker(105))
            
            # 设置亮文本颜色
            palette.setColor(QPalette.BrightText, Qt.white)
            
            # 设置链接颜色
            palette.setColor(QPalette.Link, highlight_color)
            palette.setColor(QPalette.LinkVisited, highlight_color.darker(120))
            
            # 应用调色板到主窗口
            self.setPalette(palette)
            
            # 应用调色板到QApplication（全局应用）
            QApplication.instance().setPalette(palette)
            
            # 递归应用主题到所有子组件
            self.apply_theme_to_all_widgets(self, palette)
            
            # 特殊处理工具栏
            self.apply_theme_to_toolbars(color, text_color)
            
            # 特殊处理菜单栏
            self.apply_theme_to_menubar(color, text_color)
            
            # 特殊处理状态栏
            self.apply_theme_to_statusbar(color, text_color)
            
            # 强制刷新界面
            self.update()
            
        except Exception as e:
            print(f"应用主题颜色失败: {e}")
    
    def apply_theme_to_all_widgets(self, widget, palette):
        """递归应用主题到所有子组件"""
        try:
            # 应用调色板到当前组件
            widget.setPalette(palette)
            
            # 递归处理所有子组件
            for child in widget.findChildren(QWidget):
                child.setPalette(palette)
                
                # 特殊处理不同类型的组件
                if isinstance(child, (QListWidget, QTreeWidget, QTableWidget)):
                    # 列表、树形和表格组件需要特殊处理
                    child.setPalette(palette)
                    child.setAlternatingRowColors(True)
                    
                elif isinstance(child, (QLineEdit, QTextEdit, QPlainTextEdit)):
                    # 文本输入组件
                    child.setPalette(palette)
                    
                elif isinstance(child, (QPushButton, QToolButton)):
                    # 按钮组件
                    child.setPalette(palette)
                    
                elif isinstance(child, (QComboBox, QSpinBox, QDoubleSpinBox)):
                    # 下拉框和数字输入框
                    child.setPalette(palette)
                    
                elif isinstance(child, (QScrollBar, QSlider)):
                    # 滚动条和滑块
                    child.setPalette(palette)
                    
        except Exception as e:
            print(f"应用主题到子组件失败: {e}")
    
    def apply_theme_to_toolbars(self, color, text_color):
        """应用主题到工具栏"""
        try:
            # 将text_color转换为QColor对象
            if isinstance(text_color, int):
                text_color_obj = QColor(text_color)
            else:
                text_color_obj = text_color
            
            # 获取所有工具栏
            toolbars = self.findChildren(QToolBar)
            for toolbar in toolbars:
                # 设置工具栏样式
                toolbar.setStyleSheet(f"""
                    QToolBar {{
                        background-color: {color.name()};
                        color: {text_color_obj.name()};
                        border: 1px solid {color.darker(120).name()};
                        spacing: 2px;
                    }}
                    QToolBar::separator {{
                        background-color: {color.darker(130).name()};
                        width: 1px;
                        margin: 2px;
                    }}
                    QToolButton {{
                        background-color: transparent;
                        color: {text_color_obj.name()};
                        border: none;
                        padding: 3px;
                        margin: 1px;
                    }}
                    QToolButton:hover {{
                        background-color: {color.lighter(120).name()};
                        border-radius: 3px;
                    }}
                    QToolButton:pressed {{
                        background-color: {color.darker(120).name()};
                        border-radius: 3px;
                    }}
                """)
                
        except Exception as e:
            print(f"应用主题到工具栏失败: {e}")
    
    def apply_theme_to_menubar(self, color, text_color):
        """应用主题到菜单栏"""
        try:
            # 将text_color转换为QColor对象
            if isinstance(text_color, int):
                text_color_obj = QColor(text_color)
            else:
                text_color_obj = text_color
            
            # 设置菜单栏样式
            menubar = self.menuBar()
            if menubar:
                menubar.setStyleSheet(f"""
                    QMenuBar {{
                        background-color: {color.name()};
                        color: {text_color_obj.name()};
                        border-bottom: 1px solid {color.darker(120).name()};
                    }}
                    QMenuBar::item {{
                        background-color: transparent;
                        padding: 4px 8px;
                    }}
                    QMenuBar::item:selected {{
                        background-color: {color.lighter(120).name()};
                    }}
                    QMenuBar::item:pressed {{
                        background-color: {color.darker(120).name()};
                    }}
                    QMenu {{
                        background-color: {color.name()};
                        color: {text_color_obj.name()};
                        border: 1px solid {color.darker(120).name()};
                    }}
                    QMenu::item {{
                        padding: 4px 20px;
                    }}
                    QMenu::item:selected {{
                        background-color: {color.lighter(120).name()};
                    }}
                    QMenu::separator {{
                        height: 1px;
                        background-color: {color.darker(130).name()};
                        margin: 2px 0px;
                    }}
                """)
                
        except Exception as e:
            print(f"应用主题到菜单栏失败: {e}")
    
    def apply_theme_to_statusbar(self, color, text_color):
        """应用主题到状态栏"""
        try:
            # 将text_color转换为QColor对象
            if isinstance(text_color, int):
                text_color_obj = QColor(text_color)
            else:
                text_color_obj = text_color
            
            # 设置状态栏样式
            statusbar = self.statusBar()
            if statusbar:
                statusbar.setStyleSheet(f"""
                    QStatusBar {{
                        background-color: {color.name()};
                        color: {text_color_obj.name()};
                        border-top: 1px solid {color.darker(120).name()};
                    }}
                    QStatusBar::item {{
                        border: none;
                    }}
                """)
                
        except Exception as e:
            print(f"应用主题到状态栏失败: {e}")

    def change_annotation_box_color(self):
        """更改标注框颜色"""
        try:
            # 创建颜色选择对话框
            current_color = self.line_color if hasattr(self, 'line_color') else Qt.red
            color = QColorDialog.getColor(current_color, self, "选择标注框颜色")
            
            if color.isValid():
                # 更新标注框边框颜色
                self.line_color = color
                
                # 创建基于用户选择颜色的选中填充颜色（保持透明度）
                # 使用用户选择的颜色，但保持适当的透明度以便看到底层图像
                select_fill_color = QColor(color.red(), color.green(), color.blue(), 155)
                
                # 更新Shape类的选中填充颜色（影响所有新创建的标注框）
                from libs.shape import Shape
                Shape.select_fill_color = select_fill_color
                
                # 如果有当前形状，更新其颜色
                if self.canvas.shapes:
                    for shape in self.canvas.shapes:
                        shape.line_color = color
                        # 更新每个形状的选中填充颜色
                        shape.select_fill_color = select_fill_color
                    self.canvas.update()
                
                # 保存标注框颜色设置到配置文件
                settings = QSettings()
                settings.setValue('annotation_box_color', color.name())
                settings.setValue('annotation_select_fill_color', select_fill_color.name())
                
                QMessageBox.information(self, "设置成功", f"标注框颜色已更改为: {color.name()}\n选中填充颜色已同步更新")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更改标注框颜色失败:\n{str(e)}")

    def load_settings(self):
        """加载设置"""
        try:
            settings = QSettings()
            
            # 加载主题颜色
            theme_color_name = settings.value('theme_color', None)
            if theme_color_name:
                theme_color = QColor(theme_color_name)
                if theme_color.isValid():
                    # 应用完整的主题颜色到整个应用程序
                    self.apply_theme_color(theme_color)
            
            # 加载标注框颜色
            box_color_name = settings.value('annotation_box_color', None)
            if box_color_name:
                box_color = QColor(box_color_name)
                if box_color.isValid():
                    self.line_color = box_color
            
            # 加载选中填充颜色设置
            select_fill_color_name = settings.value('annotation_select_fill_color', None)
            if select_fill_color_name:
                select_fill_color = QColor(select_fill_color_name)
                if select_fill_color.isValid():
                    # 更新Shape类的选中填充颜色
                    from libs.shape import Shape
                    Shape.select_fill_color = select_fill_color
                    
        except Exception as e:
            print(f"加载设置失败: {e}")

def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        return reader.read()
    except:
        return default


def get_main_app(argv=None):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    if not argv:
        argv = []
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(new_icon("app"))
    
    # 设置全局样式表以确保工具栏图标保持原始颜色
    app.setStyleSheet("""
        QToolBar QToolButton {
            border: none;
            background: transparent;
            padding: 3px;
        }
        QToolBar QToolButton:hover {
            background-color: rgba(0, 0, 0, 0.1);
            border-radius: 3px;
        }
        QToolBar QToolButton:pressed {
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        }
        QToolBar QToolButton:checked {
            background-color: rgba(0, 120, 215, 0.3);
            border-radius: 3px;
        }
    """)
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image_dir", nargs="?")
    argparser.add_argument("class_file",
                           default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                           nargs="?")
    argparser.add_argument("save_dir", nargs="?")
    args = argparser.parse_args(argv[1:])

    args.image_dir = args.image_dir and os.path.normpath(args.image_dir)
    args.class_file = args.class_file and os.path.normpath(args.class_file)
    args.save_dir = args.save_dir and os.path.normpath(args.save_dir)

    # Usage : labelImg.py image classFile saveDir
    win = MainWindow(args.image_dir,
                     args.class_file,
                     args.save_dir)
    win.show()
    return app, win


def main():
    """construct main app and run it"""
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())
