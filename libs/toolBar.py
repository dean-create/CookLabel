try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


class ToolBar(QToolBar):

    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def addAction(self, action):
        if isinstance(action, QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = ToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)


class ToolButton(QToolButton):
    """ToolBar companion class which ensures all buttons have the same size."""
    minSize = (60, 60)

    def minimumSizeHint(self):
        ms = super(ToolButton, self).minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        ToolButton.minSize = max(w1, w2), max(h1, h2)
        return QSize(*ToolButton.minSize)
    
    def setDefaultAction(self, action):
        """重写setDefaultAction方法以保持彩色图标的原始颜色"""
        super(ToolButton, self).setDefaultAction(action)
        
        # 检查是否是我们需要保持彩色的图标
        if action and action.icon():
            icon_name = action.objectName() if hasattr(action, 'objectName') else ''
            # 通过检查action的文本或其他属性来识别特定的图标
            action_text = action.text() if action.text() else ''
            
            # 如果是保存、复制或删除相关的动作，设置特殊样式
            colored_actions = ['save', 'copy', 'delete', '保存', '复制', '删除', 'dupBox', 'delBox']
            if any(keyword in action_text.lower() or keyword in icon_name.lower() for keyword in colored_actions):
                # 设置样式表以保持图标原始颜色
                self.setStyleSheet("""
                    QToolButton {
                        border: none;
                        background: transparent;
                    }
                    QToolButton:hover {
                        background-color: rgba(0, 0, 0, 0.1);
                        border-radius: 3px;
                    }
                    QToolButton:pressed {
                        background-color: rgba(0, 0, 0, 0.2);
                        border-radius: 3px;
                    }
                """)
