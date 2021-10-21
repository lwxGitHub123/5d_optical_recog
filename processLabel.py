import os
import cv2


def GetLabels(txt):

    labels = []
    fh = open(txt, 'r')
    imgs = []

    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        print("  line =  ")
        print(line)
        phase = words[3]
        color = words[4]
        # temp = [phase,color]
        wordss = words[3] #int(float(words[3]) * 8 / 45) * 5   #int(round(float(words[3]),0))
        print("  wordss =  ")
        print(wordss)
        temp = [words[0],words[1],words[2],words[4],wordss]
        labels.append(temp)

    return labels



def writeToTxt(content,directory,fileName):

    textPath = directory + "/" + str(fileName) + ".txt"
    f = open(textPath, 'a')

    # if imgindex % 57 == 0 :
    #     f.write('\n')
    f.write(str(content) + '\n')
    f.close()


def generateTxt():
    txtPath = "F:/liudongbo/projects/8-25/label/0902data.txt"
    labels = GetLabels(txtPath)
    for i in range(len(labels)):
         label = labels[i]
         content = str(label[0]) + '	'   + str(label[1]) + '	' + str(label[2]) + '	' + str(label[3])  + '	'+ str(label[4])
         directory = "210902"
         fileName = "0902data"

         writeToTxt(content,directory,fileName)


def readImg():

    imgPath0 = "F:/liudongbo/dataSet/bmp/210823/phase/"
    for filename in os.listdir(imgPath0):
        print(filename)
        imgPath = imgPath0 + filename
        print(imgPath)

        # 读入原始图像
        origineImage = cv2.imread(imgPath)
        print("  origineImage =  ")
        print(origineImage)

        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
        cv2.imshow(filename, origineImage)
        cv2.waitKey(0)

if __name__ == '__main__':

   # generateTxt()

   readImg()
