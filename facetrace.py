# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
import numpy as np
import os
import time
import trygrabcut

video_path='./test.mp4'
IMG='./5.jpg'
OUT_FOLDER='./out_folder'
OUT_VIDEO=''
OUT_FREQUENCY = 24 #帧输出率

class myCorrelationTracker(object):
    def __init__(self, windowName='default window', cameraNum=0):
        # 自定义几个状态标志
        self.STATUS_RUN_WITHOUT_TRACKER = 0     # 不跟踪目标，但是实时显示
        self.STATUS_RUN_WITH_TRACKER = 1    # 跟踪目标，实时显示
        self.STATUS_PAUSE = 2   # 暂停，卡在当前帧
        self.STATUS_BREAK = 3   # 退出
        self.status = self.STATUS_RUN_WITHOUT_TRACKER   # 指示状态的变量
        self.track_flag=False#是否跟踪

        # 这几个跟前面程序1定义的变量一样
        self.track_window = None  # 实时跟踪鼠标的跟踪区域
        self.drag_start = None   # 要检测的物体所在区域
        self.start_flag = True   # 标记，是否开始拖动鼠标
        self.selection = None #追加默认未选择区域
        # 创建好显示窗口
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowName, self.onMouseClicked)
        self.windowName = windowName

        # 打开摄像头
        self.cap = cv2.VideoCapture(video_path)
        self.target = cv2.imread(IMG)
        self.target = trygrabcut.call(self.target)
        
        # correlation_tracker()类，跟踪器，跟程序1中一样
        self.tracker = dlib.correlation_tracker()
        self.detector = dlib.get_frontal_face_detector()
        # 当前帧
        self.frame = None
        # 当前帧index
        self.index=0

    # 按键处理函数
    def keyEventHandler(self):
        keyValue = cv2.waitKey(5)  # 每隔5ms读取一次按键的键值
        if keyValue == 27:  # ESC
            self.status = self.STATUS_BREAK
        if keyValue == 32:  # 空格
            if self.status != self.STATUS_PAUSE:    # 按下空格，暂停播放，可以选定跟踪的区域
                #print self.status
                self.status = self.STATUS_PAUSE
                #print self.status
            else:   # 再按次空格，重新播放，但是不进行目标识别
                if self.track_window:
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True
                else:
                    self.status = self.STATUS_RUN_WITHOUT_TRACKER
        if keyValue == 13:  # 回车
            #print '**'
            if self.status == self.STATUS_PAUSE:    # 按下空格之后
                if self.track_window:   # 如果选定了区域，再按回车，表示确定选定区域为跟踪目标
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True

    # 任务处理函数        
    def processHandler(self):
        # 不跟踪目标，但是实时显示
        if self.status == self.STATUS_RUN_WITHOUT_TRACKER:
            rects=[]
            while len(rects)==0:
                ret, self.frame = self.cap.read()
                if self.frame is None:
                    self.status=self.STATUS_BREAK
                else:
                    if self.start_flag:
                        self.status=self.STATUS_PAUSE
                        rects = self.detector(self.frame, 1)
                        self.get_section(rects[0])
                        self.status = self.STATUS_RUN_WITH_TRACKER
            cv2.imshow(self.windowName, self.frame)
        # 暂停，暂停时使用鼠标拖动红框，选择目标区域，与程序1类似
        elif self.status == self.STATUS_PAUSE:
            img_first = self.frame.copy()  # 不改变原来的帧，拷贝一个新的变量出来
            if self.track_window:   # 跟踪目标的窗口画出来了，就实时标出来
                cv2.rectangle(img_first, (self.track_window[0], self.track_window[1]), (self.track_window[2], self.track_window[3]), (0,0,255), 1)
            elif self.selection:   # 跟踪目标的窗口随鼠标拖动实时显示
                cv2.rectangle(img_first, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]), (0,0,255), 1)
            cv2.imshow(self.windowName, img_first)
        # 退出
        elif self.status == self.STATUS_BREAK:
            self.cap.release()   # 释放摄像头
            if self.frame is None:
                self.save_video(OUT_FOLDER)
            cv2.destroyAllWindows()   # 释放窗口
            os._exit(1)   # 退出程序
        # 跟踪目标，实时显示
        elif self.status == self.STATUS_RUN_WITH_TRACKER:
            ret, self.frame = self.cap.read()  # 从摄像头读取一帧
            if self.frame is None:
                self.status=self.STATUS_BREAK
                return
            rects=self.detector(self.frame,1)

            if len(rects)==0:                
                if self.track_flag==False:
                    self.start_flag=True
                    self.track_flag=True
                else:
                    self.start_flag=False
            else:
                self.track_flag=False
                self.start_flag=True#更新track_window
                self.get_section(rects[0])
                
            if self.track_flag:    
                if self.start_flag:   # 如果是第一帧，需要先初始化
                    self.tracker.start_track(self.frame, dlib.rectangle(self.track_window[0], self.track_window[1], self.track_window[2], self.track_window[3]))  # 开始跟踪目标
                    self.start_flag = False   # 不再是第一帧
                else:
                    if self.frame is None:
                        self.status=self.STATUS_BREAK
                        return
                    self.tracker.update(self.frame)   # 更新

                    # 得到目标的位置，并显示
                box_predict = self.tracker.get_position()
            else:
                box_predict=rects[0]
            #cv2.rectangle(self.frame,(int(box_predict.left()),int(box_predict.top())),(int(box_predict.right()),int(box_predict.bottom())),(0,255,255),1)
            self.face_change(box_predict,self.frame)
            cv2.imshow(self.windowName, self.frame)
            save_path = "{}/{}.jpg".format(OUT_FOLDER, self.index)
            cv2.imwrite(save_path, self.frame)
            self.index += 1


    # 鼠标点击事件回调函数
    def onMouseClicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:   # 是否开始拖动鼠标，记录鼠标位置
            xMin = min(x, self.drag_start[0])
            yMin = min(y, self.drag_start[1])
            xMax = max(x, self.drag_start[0])
            yMax = max(y, self.drag_start[1])
            self.selection = (xMin, yMin, xMax, yMax)
        if event == cv2.EVENT_LBUTTONUP:   # 鼠标左键松开
            self.drag_start = None
            self.track_window = self.selection
            self.selection = None

    def get_section(self,img):
        xMin=img.left()
        yMin=img.top()
        xMax=img.right()
        yMax=img.bottom()
        self.selection = (xMin, yMin, xMax, yMax)
        self.track_window=self.selection

    def face_change(self,img,sample_image):
        x=int(img.left())
        y=int(img.top())
        w=int(img.right())-x
        h=int(img.bottom())-y
        
        temp=(0,0,0)
        #rect=cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        temp = cv2.resize(self.target,(w,h),cv2.INTER_CUBIC)
        dilate=self.cut(self.target)
        dilate=cv2.resize(dilate,(w,h),cv2.INTER_CUBIC)
        for i in range(h):
            for j in range(w):
                if dilate[i,j]==0:
                    sample_image[y+i,x+j]=temp[i,j]

    def cut(self,tar):#待定算法
        img=tar
        #日常缩放
        rows,cols,channels = img.shape#rows，cols最后一定要是前景图片的，后面遍历图片需要用到

        #转换hsv
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #获取mask
        lower_blue=np.array([0,0,221])
        upper_blue=np.array([180,30,255])#白色
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        #cv2.imshow('Mask', mask)

        #腐蚀膨胀
        erode=cv2.erode(mask,None,iterations=1)
        dilate=cv2.dilate(erode,None,iterations=1)
        #cv2.imshow('res',dilate)
        return dilate

    def save_video(self,path):
        global OUT_VIDEO
        if OUT_VIDEO!='':
            OUT_VIDEO+='/'
        filelist = os.listdir(path) #获取该目录下的所有文件名
        if len(filelist)==0:
            return
        
        height,width,layers=cv2.imread(path+"/"+filelist[0]).shape
        size=(width,height)
        print(size)
        '''
        fps:
        帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
        如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
        '''
        
        file_path = OUT_VIDEO+str(int(time.time())) + ".avi"#导出路径
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')#不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
        video = cv2.VideoWriter( file_path, fourcc, OUT_FREQUENCY, size )
        
        for item in range(1,len(filelist)+1):
            item = path +'/'+ str(item)+'.jpg'
            #print(item)
            img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)        #把图片写进视频
        video.release() #释放       

    def run(self):
        while(1):
            self.keyEventHandler()
            self.processHandler()

def ex_run(v=video_path,t=IMG,o=OUT_VIDEO):
    global video_path,IMG,OUT_VIDEO
    video_path=v
    IMG=t
    OUT_VIDEO=o
    import shutil
    try:
        shutil.rmtree(OUT_FOLDER)
    except OSError:
        pass
    
    os.mkdir(OUT_FOLDER)
    testTracker = myCorrelationTracker(windowName='image', cameraNum=1)
    testTracker.run()

##if __name__ == '__main__':
##    ex_run()
##
