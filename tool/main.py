"""
main.py
Fetal Ultrasound Facial Measurement Tool — Application Entry Point

Usage:
    python main.py
"""

import sys
import os

# Ensure the working directory is the tool/ folder so all relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from main_window import MainWindow


def main():
    # Enable high-DPI display support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
