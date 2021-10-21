# 验证模块: 加载模型 -> 分割测试图片 -> 送入模型预测 -> 测试准确率

from split import split
from txt2csv import txt2csv
from gen_dataset import gen_dataset
from train import train
from visualize import visualize
from utils import load_models
from model import CNN
import torch
################# 切割图片超参数 ###################
PIC_PATH = 'test_pic'                # 原图路径(文件夹)
PIC_NUM = 8                         # 原图数量
SAVE_PATH = 'small_test_pic'             # 切出来的小图保存路径
SUFFIX = 'tif'                      # 图片后缀,ex:'tif', 'jpg', 'png'
PX = 6                              # 切出来的小图的边宽,12卷两次池化一次变6
################# 切割图片超参数 ###################
################# txt2csv超参数 ###################
TXT_NAME = '西游记.txt'       # txt文件名
OUTPUT_FILE_NAME = 'y_test_8_20.csv'     # 输出csv名
INDEX = ['x','y','z','plr','delay'] # txt列顺序,如果是先偏振再延迟量就掉换位置
DELAY = [70, 90]                    # 延迟量可选档位(从小到大)
PLR = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]              # 偏振角可选角度(从小到大)
################# txt2csv超参数 ###################
################ 生成训练集超参数 ##################
TYPE = 'plr'                      # 要训练的模型类型
TEST_SIZE = 0.2                     # 测试集占总数据的比例
################ 生成训练集超参数 ##################
################### 训练超参数 ####################
if TYPE == 'delay':
    N_O = len(DELAY)
elif TYPE == 'plr':
    N_O = len(PLR)
elif TYPE == 'both':
    N_O = len(PLR)*len(DELAY)

################### 训练超参数 ####################
################### 画图超参数 ####################
PLOT_ONLY = 5000                    # 画多少个点在聚类图上
################### 画图超参数 ####################

print('====================== 切割图片中 ===========================')
n_split = split(pic_path=PIC_PATH, pic_num=PIC_NUM, save_path=SAVE_PATH, suffix=SUFFIX, px=PX)


print('\n====================== 生成csv中 ============================')
txt2csv(txt_name=TXT_NAME, output_file_name=OUTPUT_FILE_NAME, index=INDEX, delay=DELAY, plr=PLR)
print('\n===================== 生成训练集中 ===========================')
pic_tensor, labels, decode, (train_x, test_x, train_y, test_y) = gen_dataset(pic_path=SAVE_PATH, pic_num=n_split, filename=OUTPUT_FILE_NAME, type=TYPE, test_size=TEST_SIZE)
print('\n===================== 模型测试中 ============================')
cnn = CNN(n_o=N_O, last_layer_px=(PX*2))
load_models(cnn, TYPE)

test_output, _, _ = cnn(pic_tensor)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(type(pred_y), type(labels))
print(pred_y.shape, labels.shape)
accuracy = float((pred_y == labels.data.numpy()).astype(int).sum()) / float(pred_y.shape[0])

print('Accuracy: ', accuracy)
# print('\n===================== 可视化聚类中 ===========================')
# visualize(cnn, input=pic_tensor, label=labels, plot_only=PLOT_ONLY, n_labels=len(PLR), title='4_2-bits')


