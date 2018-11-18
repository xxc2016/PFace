# encoding=utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
import sys, cv2, time

from PyQt5.QtWidgets import QFileDialog,QMainWindow,QMessageBox

from PyQt5.QtCore import QBasicTimer, QThread, pyqtSignal, Qt

from PyQt5.QtGui import QIcon

from faceui import Ui_MainWindow
import main
import facetrace

VIDEO_PATH='123'
TARGET_IMG=''
OUT_PATH=''
FLAG=0
class mywindow(QMainWindow,Ui_MainWindow): #这个窗口继承了用QtDesignner 绘制的窗口

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.setWindowTitle('图标')
        self.setWindowIcon(QIcon('./ico.png'))
        self.pushButton_2.clicked.connect(self.get_video)
        self.pushButton.clicked.connect(self.get_image)
        self.pushButton_3.clicked.connect(self.get_dir)
        self.pushButton_4.clicked.connect(self.start)
        self.progressBar.setValue(0) 
        self.checkBox.stateChanged.connect(self.get_flag)
        self.checkBox_2.stateChanged.connect(self.change_func)
        
        self.timer = QBasicTimer()
        self.step = 0

        self.thread = MyThread() # 创建一个线程 
        self.thread.sec_changed_signal.connect(self.update) # 线程发过来的信号挂接到槽：update

        
    def get_video(self):
        if self.timer.isActive():
            return
        fileName1,ftype= QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                    "C:/",
                                    "MP4 Files (*.mp4)")   #设置文件扩展名过滤,注意用双分号间隔
        #print(fileName1)
        self.lineEdit.setText(fileName1)
        self.clear_timer()
        
    def get_image(self):
        if self.timer.isActive():
            return
        fileName1,ftype= QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                    "C:/",
                                    "JPG Files (*.jpg)")   #设置文件扩展名过滤,注意用双分号间隔
        self.lineEdit_2.setText(fileName1)
        self.clear_timer()
        
    def get_dir(self):
        if self.timer.isActive():
            return
        fileName1 = QFileDialog.getExistingDirectory(self,
                                    "选取保存地址",
                                    "C:/")   #设置文件扩展名过滤,注意用双分号间隔
        self.lineEdit_3.setText(fileName1)
        self.clear_timer()

    def get_flag(self):
        global FLAG
        if self.checkBox.isChecked():
            FLAG=1
        else:
            FLAG=0

    def change_func(self):
        global FLAG
        if self.checkBox_2.isChecked():
            self.checkBox.setCheckable(False)
            FLAG=2
        else:
            self.checkBox.setCheckable(True)
            FLAG=0
            
    def clear_timer(self):
        self.step=0
        self.progressBar.setValue(0)
        self.pushButton_4.setText('开始')
        self.timer.stop()
        self.thread.terminate()
        
    def start(self):
        global VIDEO_PATH
        global TARGET_IMG
        global OUT_PATH
        flag=1
        video=str(self.lineEdit.text())
        img=str(self.lineEdit_2.text())
        out_path=str(self.lineEdit_3.text())
        if video!='' and img!='':
            VIDEO_PATH=video
            TARGET_IMG=img
            OUT_PATH=out_path
        else:
            QMessageBox.information(self, "错误", "文件地址不能为空")
            flag=0

        if flag==1:
            self.doAction()

    def timerEvent(self, e):
        if self.step >= 100:
            self.timer.stop()
            self.label_2.setText('完成')
            return
        if self.step!=99:
            if self.step>40:
                self.step = self.step-0.5
            self.step = self.step+1
            self.progressBar.setValue(int(self.step))

    def doAction(self):

        if self.timer.isActive():
            self.timer.stop()
            self.pushButton_4.setText('开始')
            self.thread.terminate()
        else:
            self.timer.start(100, self)
            self.pushButton_4.setText('停止')
            self.thread.start()
            
    def update(self, sec):  
        self.step=sec
        self.progressBar.setValue(self.step)
        self.timer.stop()
        self.label_2.setText('完成')
        
class MyThread(QThread):  
  
    sec_changed_signal = pyqtSignal(int) # 信号类型：int
    global VIDEO_PATH
    global TARGET_IMG
    global OUT_PATH
    global FLAG
    def __init__(self, flag=1, parent=None):  
        super().__init__(parent)
        self.flag = flag # 默认1000秒
  
    def run(self):
        if self.flag==1:
            if FLAG==0 or FLAG==1:
                main.main(VIDEO_PATH,TARGET_IMG,OUT_PATH,FLAG)
            else:
                facetrace.ex_run(VIDEO_PATH,TARGET_IMG,OUT_PATH)
            self.flag=0
            self.sec_changed_signal.emit(100)  #发射信号
        else:
            return
            
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
