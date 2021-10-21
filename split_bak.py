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
        if i < n - 1 and head - order_center[i + 1][1] < 30:    # 距离小于30就算是同一行
            tmp_li.append(order_center[i])

        else:                                                   # 一行的最后一个
            tmp_li.append(order_center[i])
            res += sorted(tmp_li, key=lambda x: -x[0])          # 从大到小排
            tmp_li = []
            if i < n - 1:
                head = order_center[i + 1][1]
        i += 1
    return res



# 相邻中心点大概距离70px
def split(pic_path, pic_num, save_path, suffix='tiff', px=10):              # 一次分割两张图,延迟的和偏振的
    counter = 0                                                             # 保存的小图序号
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(1, pic_num+1):
        img_name_d = pic_path + '/' + str(i) + '_delay.' + suffix
        img_name_p = pic_path + '/' + str(i) + '_plr.' + suffix
        img_d = cv2.imread(img_name_d)                                      # 只有delay的图能分割出轮廓
        img_p = cv2.imread(img_name_p)

        gray = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)                      # 灰度处理(找轮廓的前置1)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)                         # 模糊处理(找轮廓的前置2)
        img_binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]  # 二值处理(找轮廓的前置3)
        contours = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        print('第%d张图有%d个点' % (i, len(contours)))

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


# counter = split(pic_path='210616_pic', pic_num=1, save_path='pic')

