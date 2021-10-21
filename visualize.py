# 降维可视化
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from matplotlib import cm
import matplotlib as mpl
from sklearn.manifold import TSNE
# import model
import utils
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from common import writeToTxtTime



def plot_with_labels(df, n_labels, title=''):
    # cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold", 'm', 'c', 'k', 'g', 'y', 'r',
    #                                   "RosyBrown", "IndianRed", "LightCoral", "Maroon", "DarkSalmon",
    #                                   "SeaShell", "SandyBrown", "PeachPuff","LightPink",
    #                                   "Crimson", "LavenderBlush", "PaleVioletRed", "HotPink",
    #                                   "MediumVioletRed", "Orchid", "Thistle", "plum", "Magenta",
    #                                   "Fuchsia", "DarkMagenta", "MediumOrchid", "DarkVoilet", "Indigo",
    #                                   "MediumSlateBlue", "SlateBlue", "Lavender",][:n_labels])
    cmap = mpl.colors.ListedColormap(['#F0F8FF','#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4','#000000',
'#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC',
'#DC143C','#00FFFF','#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00',
'#9932CC','#8B0000','#E9967A',][:n_labels])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_labels), n_labels)
    fig, ax = plt.subplots()

    scatter = ax.scatter(x='x', y='y', c='cluster', marker='.', data=df,
                         # norm=norm, s=100, edgecolor='none', alpha=0.70)
                             cmap=cmap, norm=norm, s=25, edgecolor='none', alpha=0.70, )
    fig.colorbar(scatter, ticks=np.linspace(0, n_labels-1, n_labels))
    plt.title(title)
    plt.xticks([])      # 去掉坐标轴
    plt.yticks([])
    plt.savefig(title+'.jpg', dpi=1000)
    plt.show()

def plot_with_labels3D(df, n_labels, title=''):
    #
    # cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold", 'm', 'c', 'k', 'g', 'y', 'r',
    #                                   "RosyBrown", "IndianRed", "LightCoral", "Maroon", "DarkSalmon",
    #                                   "SeaShell", "SandyBrown", "PeachPuff","LightPink",
    #                                   "Crimson", "LavenderBlush", "PaleVioletRed", "HotPink",
    #                                   "MediumVioletRed", "Orchid", "Thistle", "plum", "Magenta",
    #                                   "Fuchsia", "DarkMagenta", "MediumOrchid", "DarkVoilet", "Indigo",
    #                                   "MediumSlateBlue", "SlateBlue", "Lavender",][:n_labels])


    cmap = mpl.colors.ListedColormap(['#F0F8FF','#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4','#000000',
'#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC',
'#DC143C','#00FFFF','#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00',
'#9932CC','#8B0000','#E9967A',][:n_labels])


    # cmap = mpl.colors.ListedColormap(
    #     ['g','r','#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4', '#000000',
    #      '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED',
    #      '#FFF8DC',
    #      '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F',
    #      '#FF8C00',
    #      '#9932CC', '#8B0000', '#E9967A', ][:n_labels])

    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_labels), n_labels)
    fig, ax = plt.subplots()

    scatter = ax.scatter(x='x', y='y', c='cluster', marker='.', data=df,
                         # norm=norm, s=100, edgecolor='none', alpha=0.70)
                             cmap=cmap, norm=norm, s=25, edgecolor='none', alpha=0.70, )
    fig.colorbar(scatter, ticks=np.linspace(0, n_labels-1, n_labels))
    plt.title(title)
    plt.xticks([])      # 去掉坐标轴
    plt.yticks([])
    # plt.zticks([])
    plt.savefig(title+'.jpg', dpi=1000)
    plt.show()

def reduce_dimension(cnn, input, label, plot_only):    # 把CNN最后一层输出拿出来降维
    # cnn为训练好的模型,input为堆叠好的图片tensor,label为标签的tensor,plot_only为绘制的点数量
    _, last_layer, _ = cnn(input)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=63)
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
    labels = label.numpy()[:plot_only]
    df = pd.DataFrame({"x": low_dim_embs[:, 0],
                       "y": low_dim_embs[:, 1],
                       "cluster": labels
                       })
    return df

def reduce_dimension3D(cnn, input, label, plot_only):    # 把CNN最后一层输出拿出来降维
    # cnn为训练好的模型,input为堆叠好的图片tensor,label为标签的tensor,plot_only为绘制的点数量
    _, last_layer, _ = cnn(input)
    print(" last_layer.shape =  ")
    print(last_layer.shape)

    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000, random_state=63)
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
    labels = label.numpy()[:plot_only]

    print(" after last_layer.shape =  ")
    print(last_layer.shape)

    print(" low_dim_embs.shape =  ")
    print(low_dim_embs.shape)

    plot_number = 0
    for i in range(len(labels)):
        print(" i =  ** ")
        print(i)
        if i % 100 != 0 :
            np.delete(labels,i)
            # low_dim_embs.pop(i)
            np.delete(low_dim_embs, i)

        else :

            writeToTxtTime(str(labels[i]), "log", "labels-" + str(labels[i]))
            writeToTxtTime(str(low_dim_embs[i][0]) + "  " +str(low_dim_embs[i][1]) + "  " + str(low_dim_embs[i][2]), "log", "low_dim_embs-" + str(labels[i]))
            plot_number = plot_number + 1


    df = pd.DataFrame({"x": low_dim_embs[:, 0],
                       "y": low_dim_embs[:, 1],
                       "z": low_dim_embs[:, 2],
                       "cluster": labels
                       })

    fig = plt.figure(figsize=(4, 4))
    # 创建了一个figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
    plt.suptitle("Manifold Learning with %i points, %i neighbors made by liudongbo"
                 % (plot_number, 16), fontsize=14)

    print("  labels =  ")
    print(labels)

    '''绘制S曲线的3D图像'''
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], low_dim_embs[:, 2], c=labels, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)  # 初始化视角
    plt.show()

    return df


def visualize(cnn, input, label, plot_only, n_labels, title=''):
    df = reduce_dimension3D(cnn, input, label, plot_only)
    plot_with_labels3D(df, n_labels, title)
    # 保存坐标
    df.to_csv('cor.csv')

def colorVal():

    return  {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'black': '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgray': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'
    }


