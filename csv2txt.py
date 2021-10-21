import pandas as pd
import os

##########需要转换的csv文件###########
csvPath = './cor.csv'
if not os.path.exists(csvPath):
    print('Not that files:%s' % csvPath)

'''
pandas.read_csv() 报错 OSError: Initializing from file failed，
一种是函数参数为路径而非文件名称，另一种是函数参数带有中文。
'''
##########转换成txt文件###########
txtPath = './cor.txt'
data = pd.read_csv(csvPath, encoding='utf-8')

with open(txtPath, 'a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))