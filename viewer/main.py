import sys
from PyQt5 import QtWidgets
from gui import VideoProcessingApp

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = VideoProcessingApp()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
