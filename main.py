# coding=utf-8
# author xxc
# function 视频换脸
# date 2018/8/22
# 全局变量
VIDEO_PATH = './test.mp4' # 视频地址
EXTRACT_FOLDER = './extract_folder' # 存放帧图片的位置
OUT_FOLDER='./out_folder'
EXTRACT_FREQUENCY = 0.5 # 帧提取频率
OUT_FREQUENCY = 24 #帧输出率
TARGET_IMG='./7.jpg' #目标人物
scaleFactor=1.2 #人脸检测变化尺度1.0~
minNeighbors=3 #最小近邻矩阵数3~
flag=0 #原位补脸

import os
import cv2
import dlib
import time
import numpy as np
import face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sum_pic=0

def get_landmarks(im):
    rects = detector(im, 1)
    return rects

def extract_frames(video_path, dst_folder, index):
    # 主操作
    
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        os._exit(1)
    count = 1
    while True:
        _, frame = video.read()#frame每帧图片
        if frame is None:
            break
        #if count % EXTRACT_FREQUENCY == 0:
        save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
        cv2.imwrite(save_path, frame)
        index += 1
        count += 1
    #OUT_FREQUENCY=(int)(index-1/video.duration)
    video.release()
    # 打印出所提取帧的总数
    print("Totally save {:d} pics".format(index-1))
    sum_pic=index-1
    return sum_pic
    

def picvideo(path,file):
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
    
    file_path = file+str(int(time.time())) + ".avi"#导出路径
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')#不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv2.VideoWriter( file_path, fourcc, OUT_FREQUENCY, size )
    
    for item in range(1,len(filelist)+1):
        item = path + '/' + str(item)+'.jpg'
        #print(item)
        img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        video.write(img)        #把图片写进视频
    video.release() #释放

def detect(pathfrom,pathto,img):
    global flag
    filelist=os.listdir(pathfrom)
    
    target = cv2.imread(img)
    count=1
    error_cnt=0
    (x,y,w,h)=(0,0,0,0)
    temp=get_target(target)
    im2, landmarks2 = read_im_and_landmarks(img)
    for item in filelist:
        if item.endswith('.jpg'):
            imagepath=pathfrom+'/'+item
            sample_image,e =face.main(imagepath,im2,landmarks2)
            error_cnt+=e
            if e==0 or flag==1:#是否出错
                cv2.imwrite(pathto+'/'+str(count)+'.jpg', sample_image);
                count=count+1
    print(error_cnt)
    return (error_cnt)
    
def get_target(tar):
    faces =get_landmarks(tar)
    img=tar
    for index, face in enumerate(faces):
        img=tar[face.left():face.right(),face.top():face.bottom()]
    return img
            
def read_im_and_landmarks(fname):
    SCALE_FACTOR = 1
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    rects = get_landmarks(im)
    
    if len(rects) > 1:
        return im,[]
    if len(rects) == 0:
        return im,[]
    
    s=np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    return im, s

def main(v,t,file_path,f):
    # 递归删除之前存放帧图片的文件夹，并新建一个
    global flag
    flag=f
    import shutil
    try:
        shutil.rmtree(EXTRACT_FOLDER)
        shutil.rmtree(OUT_FOLDER)
    except OSError:
        pass
    
    os.mkdir(EXTRACT_FOLDER)
    os.mkdir(OUT_FOLDER)
    # 抽取帧图片，并保存到指定路径
    sum_cnt=extract_frames(v, EXTRACT_FOLDER, 1)
    error_times=detect(EXTRACT_FOLDER,OUT_FOLDER,t)
    if file_path!='':
        file_path+='/'
    picvideo(OUT_FOLDER,file_path)
    return error_times*1.0/sum_cnt


##if __name__ == '__main__':
##    main()
