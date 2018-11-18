import cv2
import numpy as np
import dlib

def call(tar):
##    img=cv2.imread('7.jpg')
    img=tar
    
    detector = dlib.get_frontal_face_detector()
    rects=detector(img,1)
    #rect=(rects[0].left(), 2*rects[0].top()-rects[0].bottom(), rects[0].right(), rects[0].bottom())
    img = img[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()]
    rect=(0,0,img.shape[0]-1,img.shape[1]-1)
    mask=np.zeros((img.shape[:2]),np.uint8)
    bgdModel=np.zeros((1,65),np.float64)
    fgdModel=np.zeros((1,65),np.float64)

    
    #cv2.imshow("kk")

    #这里计算了5次
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    #关于where函数第一个参数是条件，满足条件的话赋值为0，否则是1。如果只有第一个参数的话返回满足条件元素的坐标。
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    #mask2就是这样固定的
    #这里的img也是固定的。
    (x,y)=mask2.shape
    iMin=y
    jMin=y
    iMax=0
    jMax=0
    for i in range(x):
        for j in range(y):  
            if mask2[i,j]==0:
                img[i,j]=255
            else:
                jMin=min(j,jMin)
                jMax=max(j,jMax)
                iMin=min(i,iMin)
                iMax=max(i,iMax)
            #img=img*mask2[:,:,np.newaxis]
    #cv2.imwrite('1.jpg',img[jMin:jMax,iMin:iMax])
    return img[jMin:jMax,iMin:iMax]
