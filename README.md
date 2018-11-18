# PFace
face-p-easy：Python大作业实现视频人脸p图或替换

运行Pface.py文件，打开UI界面。
pface.py继承于faceui.py实现基于PyQt的界面
默认替换模式，包括face.py(替换模块)、main.py(视频截取与合成并调用face.py)
勾选P图模式，包括facetrace.py(启动视频并人脸追踪)、trygrabcut.py(边缘检测)
配置文件shape_predictor_68_face_landmarks.dat(dlib库提取68个人脸特征点的模型文件)

需要的库文件：dlib、cv2、PyQt、numpy、shutil
