import time
import datetime


def writeToTxtTime(content,directory,fileName):

    tt = datetime.datetime.now()
    tth = tt.strftime("%Y-%m-%d %H")

    textPath = directory + "/" + str(fileName) + "-" +str(tth) + ".txt"
    f = open(textPath, 'a')

    # if imgindex % 57 == 0 :
    #     f.write('\n')
    f.write(str(content) + '\n')
    f.close()


def writeToTxt(content,directory,fileName):

    textPath = directory + "/" + str(fileName) + ".txt"
    f = open(textPath, 'a')

    # if imgindex % 57 == 0 :
    #     f.write('\n')
    f.write(str(content) + '\n')
    f.close()

