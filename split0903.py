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


def writeToTxt(content,directory,fileName):

    textPath = directory + "/" + str(fileName) + ".txt"
    f = open(textPath, 'a')

    # if imgindex % 57 == 0 :
    #     f.write('\n')
    f.write(str(content) + '\n')
    f.close()


def saveImg(img,xmin,ymin,xmax,ymax,imgIndex,SumThreshVal,img_p,px,save_path):

    type = img.shape

    x_centor = 0
    y_centor = 0

    # imgIndex = imgIndex + 1

    if len(type) == 3 :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("gray1", cv2.WINDOW_NORMAL)
        # cv2.imshow('gray1', image)
        # 将图片二值化
        retval, img0 = cv2.threshold(image, SumThreshVal, 255, cv2.THRESH_BINARY)
        # cv2.namedWindow("binary1", cv2.WINDOW_NORMAL)
        # cv2.imshow('binary1', img0)
        roiImg = img0[ymin:ymax, xmin:xmax].copy()
        # cv2.namedWindow("roiImg", cv2.WINDOW_NORMAL)
        # cv2.imshow('roiImg', roiImg)
        x_start = 0
        x_end = 0
        y_start = 0
        y_end = 0
        colsum = np.sum(roiImg, axis=0)
        for i in range(len(colsum)):
            if colsum[i] > 0 :
                x_start = i + xmin
                # print("  x_start =  ")
                # print(x_start)
                # print("  xmin = ")
                # print(xmin)
                break

        countJ = 0
        for j in range(len(colsum) - 1, -1, -1):
            if colsum[j] > 0:
               x_end = xmax - countJ

               # print("  x_end =  ")
               # print(x_end)
               # print("  xmax = ")
               # print(xmax)
               break
            countJ = countJ + 1

        rowsum = np.sum(roiImg, axis=1)
        for i in range(len(rowsum)):
            if rowsum[i] > 0 :
                y_start = ymin + i
                # print("  y_start =  ")
                # print(y_start)
                # print("  ymin = ")
                # print(ymin)
                break

        countJ = 0
        for j in range(len(rowsum) - 1, -1, -1):
            if rowsum[j] > 0:
                y_end = ymax - countJ

                # print("  y_end =  ")
                # print(y_end)
                # print("  ymax = ")
                # print(ymax)
                break
            countJ = countJ + 1

        x_centor = int((x_start + x_end)/2)
        y_centor = int((y_start + y_end)/2)

        # print("  x_centor =  ")
        # print(x_centor)
        # print("  y_centor = ")
        # print(y_centor)
        # writeToTxt(" x_centor = " + str(x_centor) + "   y_centor =  " + str(y_centor), "log", "log")
        roiImg1 = img0[y_start:y_end, x_start:x_end]
        # cv2.namedWindow("roiImg1", cv2.WINDOW_NORMAL)
        # cv2.imshow('roiImg1', roiImg1)
        # writeToTxt(" count = " + str(imgIndex), "log", "log")
        roiImg2 = img0[y_centor - 9:y_centor + 9, x_centor - 9:x_centor + 9]
        split_img_d = img[y_centor - px:y_centor + px, x_centor - px:x_centor + px]  # 先写y后写x,切图输入[height, width]
        pic_name_d = save_path + '/' + str(imgIndex) + '_d' + ".jpg"
        cv2.imwrite(pic_name_d, split_img_d)

        split_img_p = img_p[y_centor - px:y_centor + px, x_centor - px:x_centor + px]  # 先写y后写x,切图输入[height, width]
        pic_name_p = save_path + '/' + str(imgIndex) + '_p' + ".jpg"
        cv2.imwrite(pic_name_p, split_img_p)

        # cv2.namedWindow("roiImg2", cv2.WINDOW_NORMAL)
        # cv2.imshow('roiImg2', roiImg2)
        #
        #
        # cv2.waitKey(0)
    else :
        retval, binary = cv2.threshold(img, 205, 255, cv2.THRESH_BINARY_INV)
        roiImg = binary[ymin:ymax, xmin:xmax]
        cv2.namedWindow("roiImg", cv2.WINDOW_NORMAL)
        cv2.imshow('roiImg', roiImg)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', binary)
        cv2.waitKey(0)



def  delValidEle(H,Start,End):

        ## 计算水平 或 垂直投影分割区域的总个数
        numList = []
        toatlNum = 0
        for j in range(len(Start)):
             startIndex = Start[j]
             endIndex = End[j]
             subNum = 0
             for index in range(startIndex,endIndex):
                 subNum = subNum + H[index]

             toatlNum = toatlNum + subNum
             temp = (j,subNum)
             numList.append(temp)

        ## 删除数目太少的区域
        averNum = toatlNum / len(numList)
        for i in range(len(numList)):

             tempEle = numList[i]
             # print("  tempEle=  ")
             # print(tempEle[1])
             if  tempEle[1]  < averNum / 2   or tempEle[1]  > 1.5 * averNum :
                 Start.pop(tempEle[0])
                 End.pop(tempEle[0])

        return Start,End




def splitImg(pic_path,pic_num,suffix,ThreshVal,SetExceptAreaVal,SetM,px,save_path,SumThreshVal,divVal):

    smallImgNumber = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # BASE_PATH = os.getcwd()
    # tmp_path = os.path.join(BASE_PATH, save_path)
    #
    del_file(save_path)
    AllImgNum = []
    for hi in range(1, pic_num+1):
        img_name_d = pic_path + '/' + str(hi) + '_delay.' + suffix
        img_name_p = pic_path + '/' + str(hi) + '_prl.' + suffix
        img_d = cv2.imread(img_name_d)                                      # 只有delay的图能分割出轮廓
        img_p = cv2.imread(img_name_p)

        print("  img_name_d = ")
        print(img_name_d)
        writeToTxt(str(img_name_d),"log","log")

        # cv2.namedWindow("img_p", cv2.WINDOW_NORMAL)
        # cv2.imshow('img_p', img_p)

        origineImage = img_d
        # origineImage = cv2.medianBlur(origineImage, 3)

        image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
        # cv2.imshow('gray', image)
        # 将图片二值化
        retval, img0 = cv2.threshold(image, ThreshVal, 255, cv2.THRESH_BINARY_INV)


        # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        # cv2.imshow('binary', img0)
        # 图像高与宽
        (h, w) = img0.shape
        img = img0[int(h/2)*0:int(h/1),:]

        origineimg = origineImage[int(h/2)*0 :int(h/1), :]

        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

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

            if H[i] > 0 and start == 0:
                H_Start.append(i)
                start = 1
            if H[i] <= 0 and start == 1:
                H_End.append(i)
                start = 0

        H_Start, H_End = delValidEle(H, H_Start, H_End)
        print("   len(H_Start)  =   ")
        print(len(H_Start))
        print("  len(H_End) =  ")
        print(len(H_End))

        W = getVProjection(img)
        start = 0
        W_Start = []
        W_End = []
        for i in range(len(W)):

            if W[i] > 0 and start == 0:
                W_Start.append(i)
                start = 1
            if W[i] <= 0 and start == 1:
                W_End.append(i)
                start = 0

        W_Start, W_End = delValidEle(W, W_Start, W_End)

        print("   len(W_Start)  =   ")
        print(len(W_Start))
        print("  len(W_End) =  ")
        print(len(W_End))

        # 根据分割位置获取坐标点
        for i in range(len(H_Start)):
            for j in range(len(W_Start)):
                Position.append([W_Start[j], H_Start[i], W_End[j], H_End[i]])

        # 根据确定的位置分割字符
        PositionList = []
        print(" len(Position) = ")
        print(len(Position))
        writeToTxt(" len(Position) = " + str(len(Position)), "log", "log")
        eachImgNum = 0
        divValCount = -1
        for m in range(len(Position) - 1,-1,-1):

            # Position1 = sorted(Position, reverse=True)
            # print("  m  =   ")
            # print(m)

            divValCount = divValCount + 1
           # areaVal,Position[m][0], Position[m][1], Position[m][2], Position[m][3] = saveImg(origineImage,Position[m][0], Position[m][1], Position[m][2], Position[m][3],smallImgNumber,img_p,px,save_path)
            if divValCount % divVal == 0 :

                print("  divValCount =  ")
                print(divValCount)

                # AearaSumList.append(areaVal)
                if  hi in SetExceptAreaVal :

                    if m > SetM :
                        eachImgNum =  eachImgNum + 1
                        smallImgNumber = smallImgNumber + 1
                        saveImg(origineImage, Position[m][0], Position[m][1], Position[m][2], Position[m][3], smallImgNumber,
                                SumThreshVal,img_p, px, save_path)

                        cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238), 1)

                else :
                    eachImgNum = eachImgNum + 1
                    smallImgNumber = smallImgNumber + 1
                    saveImg(origineImage, Position[m][0], Position[m][1], Position[m][2], Position[m][3], smallImgNumber,
                            SumThreshVal,img_p, px, save_path)

                    cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]),
                                  (0, 229, 238), 1)


        AllImgNum.append(eachImgNum)


            # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            # cv2.imshow('image', origineImage)
            # cv2.waitKey(0)


    return  smallImgNumber , AllImgNum






def del_file(path):
    lsdir = os.listdir(path)
    print(lsdir)
    if any(name.endswith('.py') for name in lsdir):
       print("no txt in this dir")
    else:
      print("have txt and need to remove")

    for file in lsdir:
        try:
            c_path = os.path.join(path,file)
            os.remove(c_path)
            print("rm c path: %s " % c_path)
        except:
            #del_file(path)
            os.rmdir(c_path)
            print("rm failed try again: %s " % c_path)


# 相邻中心点大概距离70px
def split(pic_path, pic_num, save_path, suffix='tiff', px=10):              # 一次分割两张图,延迟的和偏振的
    counter = 0                                                             # 保存的小图序号
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # BASE_PATH = os.getcwd()
    # tmp_path = os.path.join(BASE_PATH, save_path)
    #

    del_file(save_path)
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

