# 生成训练集
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import train_test_split    # sklearn内置分割训练集
import pandas as pd
import torchvision.transforms as transforms # (H, W, C) -> (C, H, W)
import torch
import cv2
import numpy as np
from PIL import Image
from torch.autograd import Variable
from common import writeToTxt


# 图 -> tensor
def pic2tensor(pic_path, pic_num,imgNumList):

    imgNumListSum = []
    tmpNum = 0
    for i in range(len(imgNumList)):
        tmpNum = tmpNum + imgNumList[i]
        imgNumListSum.append(tmpNum)


    # pic_path为小图存放的文件夹名, pic_num为小图数量
    delay = []
    plr = []
    decode = []
    delay_tmp = []
    plr_tmp = []
    transf = transforms.ToTensor()
    for i in range(1, pic_num+1):
        img_d = cv2.imread(pic_path + '/%s_d.jpg' % str(i))
        img_p = cv2.imread(pic_path + '/%s_p.jpg' % str(i))
        # img_d = transf(img_d).unsqueeze(0)
        # img_p = transf(img_p).unsqueeze(0)
        # img_d = img_d.numpy()
        # img_p = img_p.numpy()

        img_d = img_d.reshape((img_d.shape[2],img_d.shape[0],img_d.shape[1]))
        img_p = img_p.reshape((img_p.shape[2], img_p.shape[0], img_p.shape[1]))

        # print("  img_d.shape =  ")
        # print(img_d.shape)

        print("  i =   ")
        print(i)
        # 初始化
        # if i == 1:
        #     delay_tensor = img_d
        #     plr_tensor = img_p
        #     continue
        # 堆叠
        # delay_tensor = torch.cat((delay_tensor, img_d), 0)  # axis=0: 竖着堆叠
        # plr_tensor = torch.cat((plr_tensor, img_p), 0)      # axis=0: 竖着堆叠

        delay.append(img_d)  # axis=0: 竖着堆叠
        plr.append(img_p)  # axis=0: 竖着堆叠

        decode.append(img_p)  # axis=0: 竖着堆叠
        plr_tmp.append(img_p)  # axis=0: 竖着堆叠
        # if  i in imgNumListSum :
        #     decode = decode + delay_tmp #+ plr_tmp
        #
        #     # writeToTxt(" i = " + str(i) + "  len(decode) = " +str(len(decode))  + "  imgNumListSum = " + str(imgNumListSum) + "  len(delay_tmp) =  " + str(len(delay_tmp)) + "  len(plr_tmp) =  " + str(len(plr_tmp)), "log", "log")
        #     delay_tmp = []
        #     plr_tmp = []


    delay_tensor = torch.tensor(delay)
    delay_tensor = Variable(delay_tensor.float())
    plr_tensor = torch.tensor(plr)
    plr_tensor = Variable(plr_tensor.float())
    decode_tensor = torch.tensor(decode)
    decode_tensor = Variable(decode_tensor.float())

    delay_m = decode
    delay_m_tensor = torch.tensor(delay_m)
    delay_m_tensor = Variable(delay_m_tensor.float())


    return delay, plr ,decode ,delay_tensor, plr_tensor ,decode_tensor

# 标签 -> tensor
def csv2tensor(filename,imgNumList,divVal):
    # filename为标签csv文件名
    data = pd.read_csv(filename)
    delay_ = data['delay']
    plr_ = data['plr']
    decode_ = data['decode']

    delay = []
    plr = []
    decode = []
    for i in range(len(delay_)):
        if i % divVal == 0 :
            delay.append(delay_[i])
            plr.append(plr_[i])
            decode.append(decode_[i])

    print("delay = ")
    print(delay)
    # for i in range(len(delay)):
    #     print(" delay[i] = ")
    #     print(delay[i])


    plr_list = []
    delay_list = []
    eachImgNum = 0
    if  len(imgNumList) <= 15 :
        for i in range(len(imgNumList)):

            eachImgNum = eachImgNum + imgNumList[i]
            start = 2 * eachImgNum - imgNumList[i]
            end = 2 * eachImgNum
            print(" i = ")
            print(i)
            print(" start* = ")
            print(start)
            print(" end = ")
            print(end)
            for j in range(start,end):
                plr_list.append(plr[j])
                # writeToTxt(str(plr[j]),"log","log")
    else :
        for i in range(0,len(imgNumList)):

            eachImgNum = eachImgNum + imgNumList[i]
            if i <= 15:
                start = 2 * eachImgNum - imgNumList[i]
                end = 2 * eachImgNum
                print(" i = ")
                print(i)
                print(" start* = ")
                print(start)
                print(" end = ")
                print(end)
                for j in range(start, end):
                    plr_list.append(plr[j])



    for h in range(eachImgNum):
        delay_list.append(delay[h])
        writeToTxt(str(delay[h]), "log", "log")


    print("plr = ")
    print(plr)

    # print("decode.shape = ")
    # print(decode.shape)





    # delay = delay.to_numpy()            # df -> np -> tensor
    delay = np.array(delay)

    # delay = torch.tensor(delay)
    delay_tensor = torch.tensor(delay)
    delay_tensor = Variable(delay_tensor)

    delay_list = np.array(delay_list)
    delay_list_tensor = torch.tensor(delay_list)
    delay_list_tensor = Variable(delay_list_tensor)



    # plr = plr.to_numpy()
    plr = np.array(plr)
    # plr = torch.tensor(plr)
    plr_tensor = torch.tensor(plr)
    plr_tensor = Variable(plr_tensor)

    # plr_list = plr_list.to_numpy()
    plr_list = np.array(plr_list)
    plr_list_tensor = torch.tensor(plr_list)
    plr_list_tensor = Variable(plr_list_tensor)

    print(" eachImgNum = ")
    print(eachImgNum)

    decode_list = []
    for h in range(eachImgNum):
        decode_list.append(decode[h])

        writeToTxt(" decode_list = " + str(decode_list[h]) , "log",
                   "log")
    # decode = decode.to_numpy()
    decode = np.array(decode)
    # decode = torch.tensor(decode)
    decode_tensor = torch.tensor(decode)
    decode_tensor = Variable(decode_tensor)

    decode_list = np.array(decode_list)
    decode_list_tensor = torch.tensor(decode_list)
    decode_list_tensor = Variable(decode_list_tensor)

    return delay, plr, decode, delay_list_tensor,plr_list_tensor, decode_list_tensor         # 返回标签tensor

# 生成数据集
def gen_dataset(pic_path, pic_num, filename, type='plr', test_size=0.2,imgNumList=[],divVal = 1):
    # 前三个参数喂入之前的两个函数,type选择返回的数据集类型,test_size代表测试集占总数据的比例
    x_d, x_p,x_c,x_d_tensor,x_p_tensor ,x_c_tensor = pic2tensor(pic_path=pic_path, pic_num=pic_num,imgNumList = imgNumList)
    y_d, y_p, decode,y_d_tensor, y_p_tensor, decode_tensor = csv2tensor(filename=filename,imgNumList = imgNumList,divVal = divVal)
    # 注意标签要取和图片一样多的
    if type == 'delay':                 # 返回的标签tensor和所有图片tensor用来输入给cnn出来长向量聚类
        # 保持一样长,不一定所有标签都用
        x_c_tensor = []
        x_p_tensor = []
        x_p = []
        x_c = []
        return x_d_tensor, y_d_tensor, decode, train_test_split(x_d, y_d[:len(x_d)], test_size=test_size, random_state=14138)
    elif type == 'plr':
        x_d = []
        x_c = []
        x_d_tensor=[]
        x_c_tensor = []
        y_d = []
        y_d_tensor= []
        decode_tensor = []
        return x_p_tensor, y_p_tensor, decode, train_test_split(x_p, y_p[:len(x_p)], test_size=test_size, random_state=14138)
    elif type == 'both':                # 直接从偏振角图片中训练对应标签decode
        x_d = []
        x_p= []
        x_d_tensor = []
        y_d = []
        y_p = []
        y_d_tensor = []
        y_p_tensor = []
        return x_c_tensor, decode_tensor, decode, train_test_split(x_c, decode[:len(x_c)], test_size=test_size, random_state=14138)
