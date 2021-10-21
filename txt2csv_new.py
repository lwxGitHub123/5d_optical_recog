# 把txt里的结果转成csv保存
import pandas as pd

def txt2csv(txt_name, output_file_name, index=['x','y','z','delay','plr'], delay=[50, 70], plr=[27, 36, 45, 54]):
    data = pd.read_csv(txt_name, sep='	', header=None, names=index)

    # 替换延迟量
    for d in range(len(delay)):
        data['delay'] = data['delay'].replace(delay[d], d)
    # 替换偏振角
    for p in range(len(plr)):
        data['plr'] = data['plr'].replace(plr[p], p)


    data['decode'] = data['plr'] + data['delay'] * len(plr)      # 用来画聚类图的总标签
    data = data.iloc[:, :]
    print("  data.info() =  ")
    print(data.info())
    data.to_csv(output_file_name, index=False)
    print(data.head())


def  generateCsv():
        AllCount = 540
        classCount = 4
        # delay = [0]*540 + [1]*540
        delay = []
        for i in range(classCount):
             temp = []
             for j in range(AllCount):
                 temp.append(i)
             delay = delay + temp


        plr = []
        ## colorCount
        colorCount = 4
        for i in range(colorCount):
            plr += [i]*540

        decode = plr
        # print(plr)
        label = pd.DataFrame({'delay':delay, 'plr':plr, 'decode': decode})
        print(label.head())
        label.to_csv('y_8_27.csv')

if __name__ == '__main__':
    generateCsv()
    print()

