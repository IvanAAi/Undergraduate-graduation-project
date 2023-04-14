import sys,os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from untitled import Ui_MainWindow
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
from scipy import stats
import statistics as st
import operator
class pp(QThread):
    d = pyqtSignal(str)
    def __init__(self,id):
        super(pp, self).__init__()
        self.id = id
    def run(self):

        aa = self.id
        os.system(f"python funcMain.py -aa {aa}")
        self.d.emit("1")
class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        #继承(QMainWindow,Ui_MainWindow)父类的属性
        super(MainWindow,self).__init__()
        #初始化界面组件
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.test)
        self.pushButton_3.clicked.connect(self.close)

    def test(self):
        id = self.lineEdit.text()
        if id:
            self.p = pp(id)
            self.p.d.connect(self.refrsh)
            self.p.start()
        else:
            QMessageBox.warning(self,"提示","未输入推荐ID！",QMessageBox.Close)
    def refrsh(self):
        with open("data.txt", "r") as f:
            data = f.read().split("\n")

        #算法1
        self.textEdit.setText(data[0])
        #算法2
        self.textEdit_2.setText(data[1])
if __name__ == "__main__":
    #创建QApplication 固定写法
    app = QApplication(sys.argv)
    # 实例化界面
    window = MainWindow()
    #显示界面
    window.show()
    #阻塞
    sys.exit(app.exec_())