from math import sqrt
try:
    from libs.ustr import ustr
except ImportError:
    from ustr import ustr
import hashlib
import re
import sys

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    QT5 = True
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    QT5 = False


def new_icon(icon):
    # 对于特定的彩色图标，使用特殊处理保持原始颜色
    colored_icons = ['baocun', 'fuzhi', 'shanchu']
    
    if icon in colored_icons:
        return new_colored_icon(icon)
    else:
        # 对于其他图标，使用原始方法
        return QIcon(':/' + icon)


def new_colored_icon(icon):
    """专门用于处理需要保持彩色的图标"""
    # 直接从资源文件创建QIcon
    qicon = QIcon(':/' + icon)
    
    # 获取多个尺寸的像素图以确保在不同DPI下都能正确显示
    sizes = [16, 24, 32, 48]
    
    for size in sizes:
        # 获取指定尺寸的像素图
        pixmap = qicon.pixmap(size, size)
        
        # 确保像素图不为空
        if not pixmap.isNull():
            # 为所有状态和模式添加相同的彩色像素图
            qicon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
            qicon.addPixmap(pixmap, QIcon.Normal, QIcon.On)
            qicon.addPixmap(pixmap, QIcon.Active, QIcon.Off)
            qicon.addPixmap(pixmap, QIcon.Active, QIcon.On)
            qicon.addPixmap(pixmap, QIcon.Selected, QIcon.Off)
            qicon.addPixmap(pixmap, QIcon.Selected, QIcon.On)
            qicon.addPixmap(pixmap, QIcon.Disabled, QIcon.Off)
            qicon.addPixmap(pixmap, QIcon.Disabled, QIcon.On)
    
    return qicon


def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class Struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def format_shortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def generate_color_by_text(text):
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, 100)


def have_qstring():
    """p3/qt5 get rid of QString wrapper as py3 has native unicode str type"""
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)


# QT4 has a trimmed method, in QT5 this is called strip
if QT5:
    def trimmed(text):
        return text.strip()
else:
    def trimmed(text):
        return text.trimmed()
