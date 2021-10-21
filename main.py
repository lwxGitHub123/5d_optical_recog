# 集成各模块的主程序,原图的名字取名要用这种格式: 1_delay.tiff, 1_plr.tiff(从1开始)
from split import split
from split0903 import splitImg

from gen_dataset import gen_dataset
from train0903 import train
from visualize import visualize
from txt2csv_new import txt2csv

################# 切割图片超参数 ###################
PIC_PATH = '210902'                # 原图路径(文件夹)
PIC_NUM = 32                         # 原图数量
SAVE_PATH = 'G:/liudongbo/dataset/small_pic/'             # 切出来的小图保存路径
SUFFIX = 'tif'                      # 图片后缀,ex:'tif', 'jpg', 'png','tiff'
PX = 7                              # 切出来的小图的边宽,12卷两次池化一次变6


THRESHVAL = 200
SETEXCEPTAREAVAL = []
SETM = -1
SUMTHRESHVAL = 100



TXT_NAME = '0902data.txt'            #标签文件名
DELAY = [80,120]    #[50, 100]
PLR = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # [0,5,10, 15, 20, 25,30,35,40,45,50,55,60,65,70,75] #  #[5,20,35,50]# #   [27,36,45,54]
INDEX = ['x','y','z','delay','plr']
################# 切割图片超参数 ###################

OUTPUT_FILE_NAME = 'y_10_11.csv'

################ 生成训练集超参数 ##################
TYPE = 'both'                      # 要训练的模型类型
TEST_SIZE = 0.8                    # 测试集占总数据的比例
################ 生成训练集超参数 ##################
################### 训练超参数 ####################
if TYPE == 'delay':
    N_O = len(DELAY)
elif TYPE == 'plr':
    N_O = len(PLR)
elif TYPE == 'both':
    N_O = len(PLR)*len(DELAY)

# N_O = 4

DIVVAL = 2   #划分数据集的倍数

LR = 0.0000001
BATCH_SIZE = 32
EPOCH = 5000
################### 训练超参数 ####################
################### 画图超参数 ####################
PLOT_ONLY = 33280                   # 画多少个点在聚类图上
################### 画图超参数 ####################

print('====================== 切割图片中 ===========================')
#n_split = split(pic_path=PIC_PATH, pic_num=PIC_NUM, save_path=SAVE_PATH, suffix=SUFFIX, px=PX)
n_split,eachImgNum = splitImg(pic_path=PIC_PATH,pic_num= PIC_NUM,suffix= SUFFIX,ThreshVal=THRESHVAL,SetExceptAreaVal=SETEXCEPTAREAVAL,SetM = SETM,px=PX,save_path=SAVE_PATH,SumThreshVal=SUMTHRESHVAL,divVal=DIVVAL)
# n_split = splitByProjection(pic_path=PIC_PATH, pic_num=PIC_NUM, save_path=SAVE_PATH, suffix=SUFFIX, px=PX)
# n_split = n_split - 30


print('\n====================== 生成csv中 ============================')
txt2csv(txt_name=TXT_NAME, output_file_name=OUTPUT_FILE_NAME, index=INDEX, delay=DELAY, plr=PLR)
print('\n===================== 生成训练集中 ===========================')
pic_tensor, labels, decode, (train_x, test_x, train_y, test_y) = gen_dataset(pic_path=SAVE_PATH, pic_num=n_split, filename=OUTPUT_FILE_NAME, type=TYPE, test_size=TEST_SIZE ,imgNumList = eachImgNum,divVal = DIVVAL)
print('\n===================== 模型训练中 ============================')
cnn = train(train_x, test_x, train_y, test_y, pic_px=PX*2, n_o=N_O, lr=LR, batch_size=BATCH_SIZE, epoch=EPOCH, type=TYPE)
print('\n===================== 可视化聚类中 ===========================')
visualize(cnn, input=pic_tensor, label=labels, plot_only=PLOT_ONLY, n_labels=N_O, title='4-bits')   # title改成对应的bits


