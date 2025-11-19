def apply_dark_theme(window):
    try:
        with open("assets/themes/dark.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background: #2E2E2E;
            }
            QTabBar::tab {
                background: #2E2E2E;
                color: #FFFFFF;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                background: #3A3A3A;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QTextEdit {
                background-color: #2E2E2E;
                color: #FFFFFF;
                border: 1px solid #555;
            }
        """)

def apply_light_theme(window):
    try:
        with open("assets/themes/light.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QTabWidget::pane {
                border: 1px solid #CCC;
                background: #FFFFFF;
            }
            QTabBar::tab {
                background: #FFFFFF;
                color: #000000;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QProgressBar {
                border: 1px solid #CCC;
                border-radius: 5px;
                background: #FFFFFF;
                color: black;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QTextEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: 1px solid #CCC;
            }
        """)

def apply_gaming_theme(window):
    try:
        with open("assets/themes/gaming.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background-color: #0A0A0A;
            }
            QTabWidget::pane {
                border: 1px solid #FF5555;
                background: #1A1A1A;
            }
            QTabBar::tab {
                background: #1A1A1A;
                color: #FF5555;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #FF5555;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #FF5555;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #FF3333;
            }
            QProgressBar {
                border: 1px solid #FF5555;
                border-radius: 5px;
                background: #1A1A1A;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #FF5555;
            }
            QTextEdit {
                background-color: #1A1A1A;
                color: #FF5555;
                border: 1px solid #FF5555;
            }
        """)

def apply_crystal_theme(window):
    try:
        with open("assets/themes/crystal.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background-color: #E0F7FA;
            }
            QTabWidget::pane {
                border: 1px solid #00BCD4;
                background: #B2EBF2;
            }
            QTabBar::tab {
                background: #B2EBF2;
                color: #006064;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #00BCD4;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #00BCD4;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00ACC1;
            }
            QProgressBar {
                border: 1px solid #00BCD4;
                border-radius: 5px;
                background: #B2EBF2;
                color: #006064;
            }
            QProgressBar::chunk {
                background-color: #00BCD4;
            }
            QTextEdit {
                background-color: #B2EBF2;
                color: #006064;
                border: 1px solid #00BCD4;
            }
        """)

def apply_neon_theme(window):
    try:
        with open("assets/themes/neon.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background-color: #0D1B2A;
            }
            QTabWidget::pane {
                border: 1px solid #FF00FF;
                background: #1B263B;
            }
            QTabBar::tab {
                background: #1B263B;
                color: #FF00FF;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #FF00FF;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #FF00FF;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #E91E63;
            }
            QProgressBar {
                border: 1px solid #FF00FF;
                border-radius: 5px;
                background: #1B263B;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #FF00FF;
            }
            QTextEdit {
                background-color: #1B263B;
                color: #FF00FF;
                border: 1px solid #FF00FF;
            }
        """)

def apply_galaxy_theme(window):
    try:
        with open("assets/themes/galaxy.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background: qradialgradient(cx:0.5, cy:0.5, radius:1, fx:0.5, fy:0.5, stop:0 #0D1B2A, stop:1 #1B263B);
            }
            QTabWidget::pane {
                border: 1px solid #00D4FF;
                background: #1B263B;
            }
            QTabBar::tab {
                background: #1B263B;
                color: #00D4FF;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #00D4FF;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #00D4FF;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00B7D4;
            }
            QProgressBar {
                border: 1px solid #00D4FF;
                border-radius: 5px;
                background: #1B263B;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #00D4FF;
            }
            QTextEdit {
                background-color: #1B263B;
                color: #00D4FF;
                border: 1px solid #00D4FF;
            }
        """)

def apply_metaverse_theme(window):
    try:
        with open("assets/themes/metaverse.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1A0D2E, stop:1 #2E1A4D);
            }
            QTabWidget::pane {
                border: 1px solid #8A2BE2;
                background: #2E1A4D;
            }
            QTabBar::tab {
                background: #2E1A4D;
                color: #8A2BE2;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #8A2BE2;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #8A2BE2;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QProgressBar {
                border: 1px solid #8A2BE2;
                border-radius: 5px;
                background: #2E1A4D;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #8A2BE2;
            }
            QTextEdit {
                background-color: #2E1A4D;
                color: #8A2BE2;
                border: 1px solid #8A2BE2;
            }
        """)

def apply_quantum_theme(window):
    try:
        with open("assets/themes/quantum.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background: qradialgradient(cx:0.5, cy:0.5, radius:1, fx:0.5, fy:0.5, stop:0 #0A1A3A, stop:1 #1A2A5A);
            }
            QTabWidget::pane {
                border: 1px solid #00FFFF;
                background: #1A2A5A;
            }
            QTabBar::tab {
                background: #1A2A5A;
                color: #00FFFF;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #00FFFF;
                color: #000000;
            }
            QPushButton {
                background-color: #00FFFF;
                color: black;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00CED1;
            }
            QProgressBar {
                border: 1px solid #00FFFF;
                border-radius: 5px;
                background: #1A2A5A;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #00FFFF;
            }
            QTextEdit {
                background-color: #1A2A5A;
                color: #00FFFF;
                border: 1px solid #00FFFF;
            }
        """)

def apply_cyberpunk_theme(window):
    try:
        with open("assets/themes/cyberpunk.css", "r") as f:
            window.setStyleSheet(f.read())
    except FileNotFoundError:
        window.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1A0D2E, stop:1 #FF00FF);
            }
            QTabWidget::pane {
                border: 1px solid #FF1493;
                background: #2A0D4A;
            }
            QTabBar::tab {
                background: #2A0D4A;
                color: #FF1493;
                padding: 12px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #FF1493;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #FF1493;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #DB7093;
            }
            QProgressBar {
                border: 1px solid #FF1493;
                border-radius: 5px;
                background: #2A0D4A;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #FF1493;
            }
            QTextEdit {
                background-color: #2A0D4A;
                color: #FF1493;
                border: 1px solid #FF1493;
            }
        """)