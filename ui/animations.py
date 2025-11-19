from PyQt5.QtCore import QPropertyAnimation, QRect, QEasingCurve

def animate_tab_transition(tab_widget):
    animation = QPropertyAnimation(tab_widget, b"geometry")
    animation.setDuration(500)
    animation.setStartValue(QRect(tab_widget.x(), tab_widget.y() - 10, tab_widget.width(), tab_widget.height()))
    animation.setEndValue(QRect(tab_widget.x(), tab_widget.y(), tab_widget.width(), tab_widget.height()))
    animation.setEasingCurve(QEasingCurve.InOutQuad)
    animation.start()

def animate_button_press(button, callback):
    animation = QPropertyAnimation(button, b"geometry")
    animation.setDuration(100)
    original_geometry = button.geometry()
    animation.setStartValue(original_geometry)
    animation.setEndValue(QRect(original_geometry.x() + 2, original_geometry.y() + 2, original_geometry.width() - 4, original_geometry.height() - 4))
    animation.setEasingCurve(QEasingCurve.InOutQuad)
    animation.finished.connect(callback)
    animation.start()

def animate_3d_transition(widget):
    animation = QPropertyAnimation(widget, b"geometry")
    animation.setDuration(800)
    animation.setStartValue(QRect(widget.x() - 20, widget.y(), widget.width(), widget.height()))
    animation.setEndValue(QRect(widget.x(), widget.y(), widget.width(), widget.height()))
    animation.setEasingCurve(QEasingCurve.InOutCubic)
    animation.start()

def animate_4d_transition(widget):
    animation = QPropertyAnimation(widget, b"geometry")
    animation.setDuration(1000)
    animation.setStartValue(QRect(widget.x(), widget.y() + 30, widget.width(), widget.height()))
    animation.setEndValue(QRect(widget.x(), widget.y(), widget.width(), widget.height()))
    animation.setEasingCurve(QEasingCurve.InOutElastic)
    animation.start()