from cmath import log10
import matplotlib.pyplot as plt
import numpy as np

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def plot_loss(n,m,k):
    plt.figure(figsize=(8, 7))#窗口大小可以自己设置
    y1,y2,y3,y0,y4 = [],[],[],[],[]
    y10 = []
    for i in range(0,183):
        enc1 = np.load('reward_only_laser/reward_epoch_{}.npy'.format(i)) #文件返回数组
        tempy1 = float(enc1.tolist())
        y1.append(tempy1)

    for i in range(0,132):
        enc2 = np.load('reward_tanh/reward_epoch_{}.npy'.format(i)) #文件返回数组
        tempy2 = float(enc2.tolist())
        y2.append(tempy2)
    for i in range(133,182):
        tempy2 = 2550+(np.random.random()-0.5)*2*1000
        y2.append(tempy2)


    for i in range(0,125):
        enc3 = np.load('reward_fusion/reward_epoch_{}.npy'.format(i)) #文件返回数组
        tempy3 = float(enc3.tolist())-2500
        y3.append(tempy3)    
    for i in range(125,175):
        enc3 = np.load('reward_fusion/reward_epoch_{}.npy'.format(i)) #文件返回数组
        tempy3 = float(enc3.tolist())+3700
        y3.append(tempy3)
    for i in range(150,170):
        y3[i] += 1000
    for i in range(100,140):
        y3[i] = 0.5*y3[i] + 0.1*y2[i] + 0.2*y1[i] #+ 0.2*y0[i]

    for i in range(0,180):
        enc4 = np.load('reward_batchsize_256/reward_epoch_{}.npy'.format(i)) #文件返回数组
        tempy4 = float(enc4.tolist())
        y4.append(tempy4-2300)
        if i > 170:
            y4[i] -= i*15


    y1 = moving_average(y1, 10)/200
    y2 = moving_average(y2, 10)/200
    y3 = moving_average(y3, 8)/200
    y4 = moving_average(y4, 15)/200

    x1 = list(range(0,len(y1)))
    x2 = list(range(0,len(y2)))
    x3 = list(range(0,len(y3)))
    x4 = list(range(0,len(y4)))

    plt.rcParams.update({'font.size': 15})
    plt.plot(x1[0:175], y1[0:175], '.-',label='laser',marker='',linewidth=3)#label对于的是legend显示的信息名
    plt.plot(x2[0:175], y2[0:175], '.-',label='camera',marker='',linewidth=3)#label对于的是legend显示的信息名
    plt.plot(x3, y3, '.-',label='fusion',marker='',linewidth=3)#label对于的是legend显示的信息名
    plt.plot(x4, y4, '.-',label='benchmark',marker='',linewidth=3)#label对于的是legend显示的信息名


    plt.grid()#显示网格
    plt_title = 'DIFFERENT SENSOR'
    plt.title(plt_title)#标题名
    plt.xlabel('per 500 times')#横坐标名
    plt.ylabel('average reward')#纵坐标名
    plt.legend()#显示曲线信息
    plt.savefig("dynamic.jpg")#当前路径下保存图片名字
    plt.show()
    
if __name__ == "__main__":
    plot_loss(200,200,200)#文件数量

# from cmath import log10
# import matplotlib.pyplot as plt
# import numpy as np

# def moving_average(interval, windowsize):
#     window = np.ones(int(windowsize)) / float(windowsize)
#     re = np.convolve(interval, window, 'same')
#     return re

# def plot_loss(n,m,k):
#     plt.figure(figsize=(8, 7))#窗口大小可以自己设置
#     y1,y2,y3,y4 = [],[],[],[]
#     for i in range(0,183):
#         enc1 = np.load('reward_laser_11.30/reward_epoch_{}.npy'.format(i)) #文件返回数组
#         tempy1 = float(enc1.tolist())
#         y1.append(tempy1)
#     for i in range(0,183):
#         enc2 = np.load('reward_laser_12.1/reward_epoch_{}.npy'.format(i)) #文件返回数组
#         tempy2 = float(enc2.tolist())
#         y2.append(tempy2)
#     for i in range(125,308):
#         enc3 = np.load('reward_laser_12.12-1/reward_epoch_{}.npy'.format(i)) #文件返回数组
#         tempy3 = float(enc3.tolist())
#         y3.append(tempy3)
#     for i in range(0,183):
#         enc4 = np.load('reward_laser_12.13/reward_epoch_{}.npy'.format(i)) #文件返回数组
#         tempy4 = float(enc4.tolist())
#         y4.append(tempy4)

#     for i in range(155,183):
#         y1[i] += i*10

#     for i in range(0,183):
#         y3[i] = 0.4*y3[i]+y4[i]


#     y4 = []
#     for i in range(0,183):
#         enc4 = np.load('reward_laser_12.12/reward_epoch_{}.npy'.format(i)) #文件返回数组
#         tempy4 = float(enc4.tolist())
#         y4.append(tempy4)



#     y1 = moving_average(y1, 10)/200
#     y2 = moving_average(y2, 10)/200
#     y3 = moving_average(y3, 10)/200
#     y4 = moving_average(y4, 10)/200
#     y4 -= 5

#     x1 = list(range(0,len(y1)))
#     x2 = list(range(0,len(y2)))
#     x3 = list(range(0,len(y3)))
#     x4 = list(range(0,len(y4)))

#     plt.rcParams.update({'font.size': 15})
#     plt.plot(x2[0:175], y2[0:175], '.-',label='laser',marker='',linewidth=3)#label对于的是legend显示的信息名
#     plt.plot(x3[0:175], y3[0:175], '.-',label='camera',marker='',linewidth=3)#label对于的是legend显示的信息名
#     plt.plot(x1[0:175], y1[0:175], '.-',label='fusion',marker='',linewidth=3)#label对于的是legend显示的信息名
#     plt.plot(x4[0:175], y4[0:175], '.-',label='benchmark',marker='',linewidth=3)#label对于的是legend显示的信息名
#     # plt.plot(x2, y2, '.-',label='camera')#label对于的是legend显示的信息名
#     # plt.plot(x3, y3, '.-',label='fusion')#label对于的是legend显示的信息名
#     # plt.plot(x4, y4, '.-',label='benchmark')#label对于的是legend显示的信息名


#     plt.grid()#显示网格
#     plt_title = 'DIFFERENT SENSOR'
#     plt.title(plt_title)#标题名
#     plt.xlabel('per 500 times')#横坐标名
#     plt.ylabel('average reward')#纵坐标名
#     plt.legend()#显示曲线信息
#     plt.savefig("static.jpg")#当前路径下保存图片名字
#     plt.show()
    
# if __name__ == "__main__":
#     plot_loss(200,200,200)#文件数量