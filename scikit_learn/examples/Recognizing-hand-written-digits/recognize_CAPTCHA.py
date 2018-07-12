#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: ‘yuxuecheng‘
@contact: yuxuecheng@baicdata.com
@software: PyCharm Community Edition
@file: recognize_CAPTCHA.py
@time: 2017/4/25 上午8:54
"""

import sys
sys.path.append("/Users/yuxuecheng/PythonLib")
import os
import glob
import argparse
import time
import requests
import shutil
from PIL import Image
from sklearn import svm
from sklearn.datasets import load_svmlight_file, load_svmlight_files

from libsvm.python.svmutil import *

base_path = 'train'
DEFAULT_HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36 Core/1.47.210.400 QQBrowser/9.3.7336.400'
}


def to36(n):
    loop = '0123456789abcdefghijklmnopqrstuvwxyz'
    a = []
    while n != 0:
        a.append(loop[n % 36])
        n /= 36
    a.reverse()
    out = ''.join(a)
    return out


# 下载图片
def downloads_pic(pic_path, picname):
    # print picname

    r = int(time.time()) * 1000
    r_s = to36(int(r))
    url = "http://xxxx.com/User/Validation/" + r_s
    res = requests.get(url, stream=True)
    with open(pic_path + picname + '.jpg', 'wb') as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
        f.close()


def get_bin_table():
    threshold = 100
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return table


# 判断像素点是黑点还是白点
def get_flag(img, x, y):
    tmp_pixel = img.getpixel((x, y))
    if tmp_pixel > 228:  # 白点
        tmp_pixel = 0
    else:  # 黑点
        tmp_pixel = 1
    return tmp_pixel


# 黑点个数
def sum_9_region(img, x, y):
    width = img.width
    height = img.height
    flag = get_flag(img, x, y)
    # 如果当前点为白色区域,则不统计邻域值
    if flag == 0:
        return 0
    # 如果是黑点
    if y == 0:  # 第一行
        if x == 0:  # 左上顶点,4邻域
            # 中心点旁边3个点
            total = get_flag(img, x, y + 1) + get_flag(img, x + 1, y) + get_flag(img, x + 1, y + 1)
            return total
        elif x == width - 1:  # 右上顶点
            total = get_flag(img, x, y + 1) + get_flag(img, x - 1, y) + get_flag(img, x - 1, y + 1)
            return total
        else:  # 最上非顶点,6邻域
            total = get_flag(img, x - 1, y) + get_flag(img, x - 1, y + 1) + get_flag(img, x, y + 1) \
                    + get_flag(img, x + 1, y) \
                    + get_flag(img, x + 1, y + 1)
            return total
    elif y == height - 1:  # 最下面一行
        if x == 0:  # 左下顶点
            # 中心点旁边3个点
            total = get_flag(img, x + 1, y) + get_flag(img, x + 1, y - 1) + get_flag(img, x, y - 1)
            return total
        elif x == width - 1:  # 右下顶点
            total = get_flag(img, x, y - 1) + get_flag(img, x - 1, y) + get_flag(img, x - 1, y - 1)
            return total
        else:  # 最下非顶点,6邻域
            total = get_flag(img, x - 1, y) + get_flag(img, x + 1, y) + get_flag(img, x, y - 1) + get_flag(img, x - 1,
                                                                                                           y - 1) + get_flag(
                img, x + 1, y - 1)
            return total
    else:  # y不在边界
        if x == 0:  # 左边非顶点
            total = get_flag(img, x, y - 1) + get_flag(img, x, y + 1) + get_flag(img, x + 1, y - 1) + get_flag(img,
                                                                                                               x + 1,
                                                                                                               y) + get_flag(
                img, x + 1, y + 1)
            return total
        elif x == width - 1:  # 右边非顶点
            total = get_flag(img, x, y - 1) + get_flag(img, x, y + 1) + get_flag(img, x - 1, y - 1) + get_flag(img,
                                                                                                               x - 1,
                                                                                                               y) + get_flag(
                img, x - 1, y + 1)
            return total
        else:  # 具备9领域条件的
            total = get_flag(img, x - 1, y - 1) + get_flag(img, x - 1, y) + get_flag(img, x - 1, y + 1) + get_flag(img,
                                                                                                                   x,
                                                                                                                   y - 1) \
                    + get_flag(img, x, y + 1) + get_flag(img, x + 1, y - 1) + get_flag(img, x + 1, y) + get_flag(img,
                                                                                                                 x + 1,
                                                                                                                 y + 1)
            return total


# 分割图片
def spilt_img(img):
    # 按照图片的特点,进行切割,这个要根据具体的验证码来进行工作.
    child_img_list = []
    for index in range(4):
        x = 6 + index * 10
        y = 5
        """
        The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        """
        child_img = img.crop(box=(x, y, x + 10, img.height - 1))
        child_img_list.append(child_img)
    return child_img_list


def to_grey(img_path):
    image = Image.open(img_path)
    imgry = image.convert('L')  # 转化为灰度图
    table = get_bin_table()
    out = imgry.point(table, '1')
    return out


def grey_img(image):
    width = image.width
    height = image.height
    box = (0, 0, width, height)
    img_new = image.crop(box)
    for i in range(0, height):
        for j in range(0, width):
            num = sum_9_region(image, j, i)
            if num < 2:
                img_new.putpixel((j, i), 255)  # 设置为白色
            else:
                img_new.putpixel((j, i), 0)  # 设置为黑色
    return img_new


def get_feature(img):
    # 获取指定图片的特征值,
    # 1. 按照每排的像素点,高度为12,则有12个维度,然后为8列,总共20个维度
    # :return:一个维度为20（高度）的列表
    width, height = img.size
    pixel_cnt_list = []
    for y in range(height):
        pix_cnt_x = 0
        for x in range(width):
            if img.getpixel((x, y)) <= 100:  # 黑色点
                pix_cnt_x += 1
        pixel_cnt_list.append(pix_cnt_x)
    for x in range(width):
        pix_cnt_y = 0
        for y in range(height):
            if img.getpixel((x, y)) <= 100:  # 黑色点
                pix_cnt_y += 1
        pixel_cnt_list.append(pix_cnt_y)

    return pixel_cnt_list


def begin(pic_path, split_pic_path):
    """
    验证码灰度化并且分割图片
    :param pic_path: 待分割图片路径
    :param split_pic_path: 分割后图片路径
    :return: 
    """
    if os.path.exists(split_pic_path):
        shutil.rmtree(split_pic_path)
        # os.rmdir(split_pic_path)
    os.makedirs(split_pic_path)
    for file_name in glob.glob(pic_path + os.sep + "*.jpg"):
        # pic = Image.open(file_name)
        pic = to_grey(file_name)
        pic.save("new_code.jpg")
        pic = Image.open("new_code.jpg")
        newpic = grey_img(pic)
        childs = spilt_img(newpic)
        count = 0
        for c in childs:
            c.save(split_pic_path + os.sep + os.path.basename(file_name).split(".")[0] + "-" + str(count) + '.jpg')
            count += 1


def train(filename, merge_pic_path):
    if os.path.exists(filename):
        os.remove(filename)
    result = open(filename, 'a')
    for f in os.listdir(merge_pic_path):
        if f != '.DS_Store' and os.path.isdir(merge_pic_path + f):
            for img in os.listdir(merge_pic_path + f):
                if img.endswith(".jpg"):
                    pic = Image.open(merge_pic_path + f + "/" + img)
                    pixel_cnt_list = get_feature(pic)
                    if ord(f) >= 97:
                        line = str(ord(f)) + " "
                    else:
                        line = f + " "
                    for i in range(1, len(pixel_cnt_list) + 1):
                        line += "%d:%d " % (i, pixel_cnt_list[i - 1])
                    result.write(line + "\n")
    result.close()


def train_new(dataset_file, pic_path):
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
    result_new = open(dataset_file, 'a')
    if os.path.isfile(pic_path):
        pic = Image.open(pic_path)
        print(pic.filename)
        pixel_cnt_list = get_feature(pic)
        line = "0 "
        for i in range(1, len(pixel_cnt_list) + 1):
            line += "%d:%d " % (i, pixel_cnt_list[i - 1])
        result_new.write(line + "\n")
    else:
        for f in glob.glob(pic_path + os.sep + "*.jpg"):
            pic = Image.open(f)
            print(pic.filename)
            pixel_cnt_list = get_feature(pic)
            line = "0 "
            for i in range(1, len(pixel_cnt_list) + 1):
                line += "%d:%d " % (i, pixel_cnt_list[i - 1])
            result_new.write(line + "\n")
    result_new.close()


# 模型训练
def train_svm_model(data_file, model_file):
    y, x = svm_read_problem(data_file)
    trained_model = svm_train(y, x)
    # svm_save_model(base_path + os.sep + "svm_model_file", model)
    svm_save_model(model_file_name=model_file, model=trained_model)


# 使用测试集测试模型
def svm_model_test(data_filename, model_filename):
    """
    svm_read_problem函数读取filename指定文件中的数据，文件内容的格式为   标签 数据
    举例如下：
    0 1:0 2:10 3:5 4:3 5:4 6:4 7:4 8:3 9:4 10:9 11:5 12:0 13:0 14:0 15:0 16:0 17:9 18:7 19:3 20:4 21:7 22:8 23:5 24:2 25:2 26:4 
    0 1:5 2:10 3:3 4:3 5:3 6:2 7:2 8:7 9:4 10:6 11:0 12:0 13:0 14:0 15:0 16:0 17:4 18:5 19:6 20:9 21:6 22:6 23:4 24:1 25:1 26:3 
    0 1:3 2:10 3:4 4:4 5:3 6:2 7:2 8:4 9:6 10:1 11:1 12:1 13:1 14:1 15:2 16:2 17:4 18:2 19:3 20:4 21:7 22:8 23:13 24:2 25:1 26:3 
    0 1:0 2:10 3:0 4:0 5:0 6:1 7:7 8:2 9:2 10:3 11:2 12:3 13:2 14:2 15:1 16:1 17:5 18:4 19:4 20:5 21:5 22:5 23:5 24:1 25:1 26:1 
    0 1:0 2:5 3:1 4:1 5:4 6:4 7:2 8:3 9:3 10:5 11:4 12:1 13:1 14:0 15:0 16:9 17:8 18:4 19:5 20:4 21:6 22:6 23:4 24:1 25:1 26:4 
    0 1:0 2:0 3:0 4:0 5:2 6:5 7:4 8:4 9:4 10:4 11:4 12:3 13:3 14:5 15:4 16:10 17:8 18:9 19:6 20:4 21:8 22:8 23:4 24:1 25:1 26:3 
    0 1:3 2:5 3:4 4:5 5:3 6:7 7:2 8:2 9:3 10:0 11:0 12:0 13:0 14:0 15:0 16:4 17:4 18:4 19:4 20:3 21:9 22:9 23:1 24:0 25:1 26:3 
    
    yt是一个list
    [0.0, 0.0, 0.0, 0.0, 0.0]
    
    xt是一个dict组成的字典
    [{1: 0.0, 2: 10.0, 3: 5.0, 4: 3.0, 5: 4.0, 6: 4.0, 7: 4.0, 8: 3.0, 9: 4.0, 10: 9.0, 11: 5.0, 12: 0.0, 13: 0.0, 
      14: 0.0, 15: 0.0, 16: 0.0, 17: 9.0, 18: 7.0, 19: 3.0, 20: 4.0, 21: 7.0, 22: 8.0, 23: 5.0, 24: 2.0, 25: 2.0, 26: 4.0}, 
     {1: 5.0, 2: 10.0, 3: 3.0, 4: 3.0, 5: 3.0, 6: 2.0, 7: 2.0, 8: 7.0, 9: 4.0, 10: 6.0, 11: 0.0, 12: 0.0, 13: 0.0, 
      14: 0.0, 15: 0.0, 16: 0.0, 17: 4.0, 18: 5.0, 19: 6.0, 20: 9.0, 21: 6.0, 22: 6.0, 23: 4.0, 24: 1.0, 25: 1.0, 26: 3.0}, 
     {1: 3.0, 2: 10.0, 3: 4.0, 4: 4.0, 5: 3.0, 6: 2.0, 7: 2.0, 8: 4.0, 9: 6.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 
      14: 1.0, 15: 2.0, 16: 2.0, 17: 4.0, 18: 2.0, 19: 3.0, 20: 4.0, 21: 7.0, 22: 8.0, 23: 13.0, 24: 2.0, 25: 1.0, 26: 3.0}, 
     {1: 0.0, 2: 10.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 7.0, 8: 2.0, 9: 2.0, 10: 3.0, 11: 2.0, 12: 3.0, 13: 2.0, 
      14: 2.0, 15: 1.0, 16: 1.0, 17: 5.0, 18: 4.0, 19: 4.0, 20: 5.0, 21: 5.0, 22: 5.0, 23: 5.0, 24: 1.0, 25: 1.0, 26: 1.0}, 
     {1: 0.0, 2: 5.0, 3: 1.0, 4: 1.0, 5: 4.0, 6: 4.0, 7: 2.0, 8: 3.0, 9: 3.0, 10: 5.0, 11: 4.0, 12: 1.0, 13: 1.0, 
     14: 0.0, 15: 0.0, 16: 9.0, 17: 8.0, 18: 4.0, 19: 5.0, 20: 4.0, 21: 6.0, 22: 6.0, 23: 4.0, 24: 1.0, 25: 1.0, 26: 4.0}]
    
    :param filename: 测试集文件名
    :return: 识别出来的数字列表
    """

    yt, xt = svm_read_problem(data_filename)
    model = svm_load_model(model_filename)
    p_label, p_acc, p_val = svm_predict(yt, xt, model)  # p_label即为识别的结果
    cnt = 0
    results = []
    result = ''
    for item in p_label:  # item:float
        if int(item) >= 97:
            result += chr(int(item))
        else:
            result += str(int(item))
        cnt += 1
        if cnt % 4 == 0:
            results.append(result)
            result = ''
    return results


def test_libsvm(train_data, test_data):
    y, x = svm_read_problem(train_data)
    yt, xt = svm_read_problem(test_data)
    trained_model = svm_train(y, x)
    print("use lib svm")
    p_label, p_acc, p_val = svm_predict(yt, xt, trained_model)  # p_label即为识别的结果
    cnt = 0
    results = []
    result = ''
    for item in p_label:  # item:float
        if int(item) >= 97:
            result += chr(int(item))
        else:
            result += str(int(item))
        cnt += 1
        if cnt % 4 == 0:
            results.append(result)
            result = ''
    return results


def test_sklearn_svm(train_data, test_data):
    print("use sklearn svm")
    X_train, Y_train, X_test, Y_test = load_svmlight_files((train_data, test_data))
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognize the CAPTCHA.")
    """
    The prepare job will generate the data set（svm_test） from the picture
    The train job will generate svm_model_file from svm_test generated by prepare job
    The test job will use the svm model file generated by train job to predict the result for the data set(svm_test) 
    generated by prepare job
    """
    # parser.add_argument('--job', choices=['prepare', 'train', 'test'], required=True, help='The job you want to execute.')
    """
    Create a mutually exclusive group. argparse will make sure that only one of the arguments in the mutually 
    exclusive group was present on the command line:
    The add_mutually_exclusive_group() method also accepts a required argument, to indicate that at least one 
    of the mutually exclusive arguments is required:
    """
    # group = parser.add_mutually_exclusive_group(required=False)
    # group.add_argument('--dir', required=False, help='The directory of the data')
    # group.add_argument('--file', required=False, help='The file of the data')

    # args = parser.parse_args()
    # job = args.job
    # directory = args.dir
    # file_name = args.file
    # data = directory if directory is not None else file_name
    # print(data)
    # if job == 'prepare':
    #     begin("data_new", "split_data")
    #     train_new(base_path + os.sep + "svm_test", "split_data")
    # elif job == 'train':
    #     train_svm_model("svm_test")
    # elif job == 'test':
    #     results_bak = svm_model_test("svm_test")
    #     print(results_bak)

    """
    following use subparsers
    """
    subparsers = parser.add_subparsers(help='additional help message',
                                       title='sub commands',
                                       description='valid sub commands',
                                       dest='sub_parser_name')
    # create the parser for the "prepare" command
    parser_prepare = subparsers.add_parser("prepare", help="prepare the data set")
    parser_prepare.add_argument("-o", "--original", required=True, help="original pictures data directory")
    parser_prepare.add_argument("-s", "--split", required=True, help="the directory used to save split pictures")
    parser_prepare.add_argument("-d", "--data", required=True, help="the file name used to save data set")

    parser_prepare = subparsers.add_parser("train", help="train the svm model")
    parser_prepare.add_argument("-d", "--data", required=True, help="the file name of the train data")
    parser_prepare.add_argument("-m", "--model", required=True, help="the file name used to save trained model")

    parser_prepare = subparsers.add_parser("test", help="test the data set")
    parser_prepare.add_argument("-d", "--data", required=True, help="the file name of the test data")
    parser_prepare.add_argument("-m", "--model", required=True, help="the file name of saved trained model")

    parser_prepare = subparsers.add_parser("libsvm", help="use libsvm to classify")
    parser_prepare.add_argument("--train", required=True, help="the file name of the train data")
    parser_prepare.add_argument("--test", required=True, help="the file name of the test data")

    parser_prepare = subparsers.add_parser("sklearn", help="use sklearn svm to classify")
    parser_prepare.add_argument("--train", required=True, help="the file name of the train data")
    parser_prepare.add_argument("--test", required=True, help="the file name of the test data")

    args = parser.parse_args()
    print(args)
    sub_command = args.sub_parser_name
    if sub_command == 'prepare':
        picture_dir = args.original
        split_dir = args.split
        data_file = args.data
        print("\t".join([picture_dir, split_dir, data_file]))
        begin(pic_path=picture_dir, split_pic_path=split_dir)
        train_new(dataset_file=data_file, pic_path=split_dir)
    elif sub_command == 'train':
        data_file = args.data
        model_file = args.model
        print("\t".join([data_file, model_file]))
        train_svm_model(data_file, model_file)
    elif sub_command == 'test':
        data_file = args.data
        model_file = args.model
        print("\t".join([data_file, model_file]))
        results = svm_model_test(data_file, model_file)
        print(results)
    elif sub_command == 'libsvm':
        train_data = args.train
        test_data = args.test
        print("\t".join([train_data, test_data]))
        results = test_libsvm(train_data, test_data)
        print(results)
    elif sub_command == 'sklearn':
        train_data = args.train
        test_data = args.test
        print("\t".join([train_data, test_data]))
        results = test_sklearn_svm(train_data, test_data)
        print(results)
