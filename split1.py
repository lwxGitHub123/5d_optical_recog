# 分割图片到文件夹中,大的图变成小的光点图
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



'''水平投影'''


def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 0:
                h_[y] += 1
    # 绘制水平投影图像
    # for y in range(h):
    #     for x in range(h_[y]):
    #         hProjection[y, x] = 255
    # cv2.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 0:
                w_[x]+=1
    #绘制垂直平投影图像
    # for x in range(w):
    #     # for y in range(h-w_[x],h):
    #     for y in range( w_[x]):
    #         vProjection[y,x] = 255
    # cv2.imshow('vProjection',vProjection)
    return w_



# 排序模块,用来把中心点从右下到左上排序
def sort_li(li):
    # 输入一个列表[[x1, y1], [x2, y2], ..., ], 输出有序列表, x,y都从大到小排
    center_list = []
    for c in li:
        x_tmp = int(np.mean([x for x in c[:, :, 0]]))
        y_tmp = int(np.mean([x for x in c[:, :, 1]]))
        center_list.append([x_tmp, y_tmp])
    order_center = sorted(center_list, key=lambda x: -x[1])     # 倒序,从大到小
    res = []
    n = len(order_center)
    i = 0
    head = order_center[0][1]                                   # 这一行内最大的y
    tmp_li = []
    while i < n:
        if i < n - 1 and head - order_center[i + 1][1] < 10:    # 距离小于10就算是同一行
            tmp_li.append(order_center[i])

        else:                                                   # 一行的最后一个
            tmp_li.append(order_center[i])
            res += sorted(tmp_li, key=lambda x: -x[0])          # 从大到小排
            tmp_li = []
            if i < n - 1:
                head = order_center[i + 1][1]
        i += 1
    return res


def split(pic_path, pic_num, save_path, suffix='tif', px=7):                # 一次分割两张图,延迟的和偏振的
    counter = 0                                                             # 保存的小图序号
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(1, pic_num+1):
        img_name_d = pic_path + '/' + str(i) + '_delay.' + suffix
        img_name_p = pic_path + '/' + str(i) + '_plr.' + suffix
        img_d = cv2.imread(img_name_d)                                      # 只有delay的图能分割出轮廓
        img_p = cv2.imread(img_name_p)

        # cv2.namedWindow("img_d", cv2.WINDOW_NORMAL)
        # cv2.imshow('img_d', img_d)
        # cv2.namedWindow("img_p", cv2.WINDOW_NORMAL)
        # cv2.imshow('img_p', img_p)
        # cv2.waitKey(0)

        gray = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)                      # 灰度处理(找轮廓的前置1)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)                         # 模糊处理(找轮廓的前置2)
        img_binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]  # 二值处理(找轮廓的前置3)
        contours = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        print('第%d张图有%d个点' % (i, len(contours)))
        # print(contours)

        # 对中心点排序
        res = sort_li(contours)
        ####################### 图像分割 ######################
        for (x, y) in res:
            counter += 1
            pic_name_d = save_path + '/' + str(counter) + '_d' + ".jpg"
            split_img_d = img_d[y - px:y + px, x - px:x + px]   # 先写y后写x,切图输入[height, width]
            cv2.imwrite(pic_name_d, split_img_d)

            pic_name_p = save_path + '/' + str(counter) + '_p' + ".jpg"
            split_img_p = img_p[y - px:y + px, x - px:x + px]   # 先写y后写x
            cv2.imwrite(pic_name_p, split_img_p)
        ####################### 图像分割 ######################
    return counter                                              # 返回切出来多少图

def splitByProjection(pic_path, pic_num, save_path, suffix='tif', px=7):

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(1, pic_num+1):
        img_name_d = pic_path + '/' + str(i) + '_delay.' + suffix
        img_name_p = pic_path + '/' + str(i) + '_plr.' + suffix
        img_d = cv2.imread(img_name_d)                                      # 只有delay的图能分割出轮廓
        img_p = cv2.imread(img_name_p)

    origineImage = img_d
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.imshow('gray', image)
    # 将图片二值化
    retval, img0 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)


    # cv2.imshow('binary', img)
    # 图像高与宽
    (h, w) = img0.shape
    img = img0[int(h/2)*0:int(h/1),:]

    origineimg = origineImage[int(h/2)*0 :int(h/1), :]

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    cv2.waitKey(0)

    Position = []
    # 水平投影
    H = getHProjection(img)

    print("  len(H) =  ")
    print(len(H))

    # cv2.waitKey(0)

    start = 0
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(H)):
        # print(" i = ")
        # print(i)
        # print("  H[i] = ")
        # print( H[i])

        if H[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0

    print("   len(H_Start)  =   ")
    print(len(H_Start))
    print("  len(H_End) =  ")
    print(len(H_End))

    W_Start_Len =  -1
    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        # 获取行图像
        # cropImg = img[H_Start[i]:H_End[i], 0:w]
        # cropImgi = origineImage[H_Start[i]:H_End[i], 0:w]
        # cv2.imshow('cropImg' + str(i),cropImgi)
        # cv2.waitKey(0)

        # 对行图像进行垂直投影
        W = getVProjection(img)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        # print(" len(W) =  ")
        # print(len(W))
        for j in range(len(W)):
            # print(" W[j] == ")
            # print(W[j])
            # print(" j == ")
            # print(j)
            if W[j] > 0 and Wstart == 0:
                W_Start = j
                Wstart = 1
                Wend = 0
            if W[j] <= 0 and Wstart == 1:
                W_End = j
                Wstart = 0
                Wend = 1
            if Wend == 1:
               if j % (len(W)) == 0 :
                  Position.append([W_Start, H_Start[i], W_End, H_End[i],1])
               else :
                  Position.append([W_Start, H_Start[i], W_End, H_End[i], 0])
               Wend = 0

        W_Len = len(W)

    # 根据确定的位置分割字符
    PositionList = []
    counter = 0
    print(" len(Position) = ")
    print(len(Position))
    # for m in range(len(Position) - 1,-1,-1):
    for m in range(len(Position)):


        # Position1 = sorted(Position, reverse=True)
        # print("  m  =   ")
        # print(m)


        # areaVal = calSum(origineimg,Position[m][0], Position[m][1], Position[m][2], Position[m][3],m)
        # areaVal = calSum(origineImage, Position[m][0], Position[m][1], Position[m][2], Position[m][3], m)
        roiImg = img[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]

        counter += 1
        pic_name_d = save_path + '/' + str(counter) + '_d' + ".jpg"
        split_img_d = img_d[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]  # 先写y后写x,切图输入[height, width]
        dim = (2*px, 2*px)
        resize_d = cv2.resize(split_img_d, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(pic_name_d, resize_d)

        pic_name_p = save_path + '/' + str(counter) + '_p' + ".jpg"
        split_img_p = img_p[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]  # 先写y后写x
        resize_p = cv2.resize(split_img_p, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(pic_name_p, resize_p)

        # AearaSumList.append(areaVal)
        # Position[m].append(areaVal)
        #
        # PositionList.append(Position[m])
        #
        #
        # cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238), 1)
        #
    return counter





# counter = split(pic_path='origin_pic', pic_num=2, save_path='small_pic')

