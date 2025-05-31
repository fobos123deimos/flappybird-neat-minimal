from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class UiMainWindow(object):
    """
    PyQt5 UI class for displaying the current score of the Flappy Bird agent.
    """

    def setupUi(self, MainWindow):
        """
        Configures and sets up the main window interface.
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(215, 120)
        MainWindow.setStyleSheet("background-color: rgb(85, 255, 255);")

        self.central_widget = QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")

        self.lcd_display = QLCDNumber(self.central_widget)
        self.lcd_display.setGeometry(QRect(10, 30, 201, 41))
        self.lcd_display.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lcd_display.setObjectName("lcd_display")
        self.lcd_display.setNumDigits(7)

        self.label = QLabel(self.central_widget)
        self.label.setGeometry(QRect(10, 10, 47, 13))
        self.label.setStyleSheet("font: 75 11pt 'Roboto';")
        self.label.setObjectName("label")

        MainWindow.setCentralWidget(self.central_widget)

        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 215, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.show()

    def retranslateUi(self, MainWindow):
        """
        Sets the window title and label text.
        """
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Flappy NEAT Score Display"))
        self.label.setText(_translate("MainWindow", "Score:"))

    def update_score(self, score):
        """
        Updates the LCD display with the current score.

        Parameters:
            score (int or float): The score value to display.
        """
        self.lcd_display.display(str(score))
