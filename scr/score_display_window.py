from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class UiMainWindow(object):
    """
    PyQt5 UI class for displaying the current score of the Flappy Bird agent.
    This interface features an LCD-style counter and a label, designed for use
    in NEAT-based experiments with Flappy Bird.
    """

    def setupUi(self, MainWindow):
        """
        Initializes and configures the main window components.

        Parameters:
            MainWindow (QMainWindow): The main application window to which UI components will be attached.
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(215, 120)
        MainWindow.setStyleSheet("background-color: rgb(85, 255, 255);")

        # Create and configure central widget
        self.central_widget = QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")

        # Create and configure LCD display for the score
        self.lcd_display = QLCDNumber(self.central_widget)
        self.lcd_display.setGeometry(QRect(10, 30, 201, 41))
        self.lcd_display.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lcd_display.setObjectName("lcd_display")
        self.lcd_display.setNumDigits(7)

        # Create and configure the label above the LCD
        self.label = QLabel(self.central_widget)
        self.label.setGeometry(QRect(10, 10, 47, 13))
        self.label.setStyleSheet("font: 75 11pt 'Roboto';")
        self.label.setObjectName("label")

        # Set central widget in the main window
        MainWindow.setCentralWidget(self.central_widget)

        # Create and configure menu bar
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 215, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        # Create and configure status bar
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Set textual content and finalize connection
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

        # Show the main window
        MainWindow.show()

    def retranslateUi(self, MainWindow):
        """
        Sets the textual content for UI elements such as the title and labels.

        Parameters:
            MainWindow (QMainWindow): The main application window.
        """
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Flappy NEAT Score Display"))
        self.label.setText(_translate("MainWindow", "Score:"))

    def update_score(self, score):
        """
        Updates the LCD display with the current score.

        Parameters:
            score (int or float): The score value to be shown on the screen.
        """
        self.lcd_display.display(str(score))
