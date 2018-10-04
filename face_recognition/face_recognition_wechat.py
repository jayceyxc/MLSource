#!/usr/bin/env python3

"""
@author: yuxuecheng
@contact: yuxuecheng@xinluomed.com
@software: PyCharm
@file: face_recognition_wechat.py
@time: 2018/8/13 1:07 PM
"""


# 人脸识别

import cv2
import dlib
import face_recognition
import facenet.detect_face
import facenet.detect_face_fromfacenet
import tensorflow as tf
import numpy as np
import os
import re
from sklearn import neighbors
import sklearn

# 图像采集
def get_image(img_path=None):
    if img_path==None:
        # 1、调用摄像头进行拍照
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        cap.release()
    else:
        # 2、根据提供的路径读取图像
        img=cv2.imread(img_path)

    return img

# 人脸检测，如果有多张人脸，返回人脸最大的那一张
def face_check(img,alg):

    dets=None

    if alg=='opencv':
        # 1、使用 opencv 检测人脸
        # 加载人脸检测分类器（正面人脸），位于OpenCV的安装目录下
        face_cascade=cv2.CascadeClassifier('/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        # 转灰度图
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 检测人脸（可能有多张），返回人脸位置信息(x,y,w,h)
        img_faces=face_cascade.detectMultiScale(img_gray)

    elif alg=='dlib':
        # 2、使用 Dlib 检测人脸
        # 安装 Dlib
        # conda activate tensorflow
        # conda install -c menpo dlib
        detector = dlib.get_frontal_face_detector()
        dets = detector(img, 1)

        img_faces=[]
        for i in range(len(dets)):
            x = dlib.rectangle.left(dets[i])
            y = dlib.rectangle.top(dets[i])
            h = dlib.rectangle.height(dets[i])
            w = dlib.rectangle.width(dets[i])
            img_faces.append([x,y,w,h])

    elif alg=='facerecognition':
        # 3、使用 face_recognition 检测人脸
        # 安装 face_recognition
        # 需要先安装dlib , 还有 CMake ( sudo apt-get install cmake )
        # conda activate tensorflow
        # pip install face_recognition

        face_locations = face_recognition.face_locations(img)

        img_faces = []
        for i in range(len(face_locations)):
            x = face_locations[i][3]
            y = face_locations[i][0]
            h = face_locations[i][2] - face_locations[i][0]
            w = face_locations[i][1] - face_locations[i][3]
            img_faces.append([x, y, w, h])

    elif alg=='facenet':
        # 4、使用 FaceNet 检测人脸
        # 安装 FaceNet
        # 到FaceNet的github上将源代码下载下来，以上相应的模型 https://github.com/davidsandberg/facenet
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                pnet, rnet, onet = facenet.detect_face_fromfacenet.create_mtcnn(sess, './facenet/model_check_point/')

        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709

        bounding_boxes, _ = facenet.detect_face_fromfacenet.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        img_faces = []
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)

            x = face_position[0]
            y = face_position[1]
            h = face_position[3] - face_position[1]
            w = face_position[2] - face_position[0]
            img_faces.append([x, y, w, h])

    # 获取面积最大的人脸
    face_xywh=[]
    for (x,y,w,h) in img_faces:
        face_xywh.append([w*h,h,w,y,x])

    max_face=[]
    face_num=len(face_xywh)
    if face_num>0:
        face_xywh=sorted(face_xywh,reverse=True)
        x=face_xywh[0][4]
        y=face_xywh[0][3]
        w=face_xywh[0][2]
        h=face_xywh[0][1]

        max_face=img[y:y+h,x:x+w]

    return max_face,dets


# 获取模型路径，用于 FaceNet
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


# 提取人脸特征
def get_feature(face_img,alg,dets=None):
    if alg=='opencv' or alg=='dlib':
        # 1、dlib 提取人脸特征
        # opencv 无法直接提取人脸特征，在这里设置 opencv 也采用 dlib 的特征提取方式
        # 下载模型：http://dlib.net/files/
        # 下载文件：shape_predictor_68_face_landmarks.dat.bz2
        # 解压文件，得到 shape_predictor_68_face_landmarks.dat 文件
        # 获取人脸检测器
        predictor = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
        for index,face in enumerate(dets):
            face_feature = predictor(face_img,face)
    elif alg=='facerecognition':
        # 2、face_recognition 提取人脸特征
        face_feature = face_recognition.face_encodings(face_img)
        face_feature=face_feature[0]
    elif alg=='facenet':
        # 3、FaceNet 提取人脸特征
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                batch_size=None
                image_size=160
                images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                                       image_size,
                                                                       image_size, 3), name='input')

                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

                model_checkpoint_path = './facenet/model_check_point/'
                input_map = {'input': images_placeholder, 'phase_train': phase_train_placeholder}

                model_exp = os.path.expanduser(model_checkpoint_path)
                meta_file, ckpt_file = get_model_filenames(model_exp)

                saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
                saver.restore(sess, os.path.join(model_exp, ckpt_file))

                face_img = cv2.resize(face_img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
                data = np.stack([face_img])

                #images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                feed_dict = {images_placeholder: data, phase_train_placeholder: False}
                face_feature = sess.run(embeddings, feed_dict=feed_dict)
                face_feature=face_feature[0]

    return face_feature

# 获取所有人像照片的特征
def get_features_labels(img_dir,alg):
    features=[]
    labels=[]

    for filePath in os.listdir(img_dir):
        file_name = filePath.split('/')[-1]
        labels.append(file_name)

        img=get_image(img_dir+filePath)

        face_img,dets=face_check(img,alg)

        if alg == 'dlib' or alg == 'opencv':
            # 由于 dlib 的人脸特征提取函数只能传入原图像和人脸位置信息，因而特殊处理
            face_img = img
        face_feature=get_feature(face_img,alg,dets)

        features.append(face_feature)

    return features,labels

# 人脸识别，返回人像的姓名
def get_name(face_feature,features,labels,dis_alg):

    name=''

    if dis_alg=='eucl':
        # 1、欧氏距离
        min_dis=99999
        min_idx=-1
        for i in range(len(features)):
            dis=np.sqrt(np.sum(np.square(face_feature-features[i])))
            if dis<min_dis:
                min_dis=dis
                min_idx=i
        name=labels[min_idx]

    elif dis_alg=='knn':
        # 2、KNN
        knn = neighbors.KNeighborsClassifier(n_neighbors=2)
        knn.fit(features, labels)
        name = knn.predict([face_feature])
        name = name[0]

    return name


if __name__ == "__main__":
    # 设置采用的算法，opencv dlib facerecognition facenet
    alg='facerecognition'

    # 1、采集图像
    img=get_image('/data/work/tensorflow/data/face_test/20.jpg')

    # 2、人脸检测
    face_img,dets=face_check(img,alg)

    # 3、预处理
    # 根据实际采集图像的质量情况进行处理

    # 4、特征提取
    if alg=='dlib' or alg=='opencv':
        # 由于 dlib 的人脸特征提取函数只能传入原图像和人脸位置信息，因而特殊处理
        face_img=img
    face_feature=get_feature(face_img,alg,dets)

    # 5、匹配识别
    # 设置距离算法，eucl knn
    dis_alg='eucl'
    # 获取图片库中的特征和标签
    features,labels=get_features_labels('/data/work/tensorflow/data/face_test/rec_test/',alg)
    # 匹配识别，获取人像的姓名
    name=get_name(face_feature,features,labels,dis_alg)

    print('识别结果：',name)