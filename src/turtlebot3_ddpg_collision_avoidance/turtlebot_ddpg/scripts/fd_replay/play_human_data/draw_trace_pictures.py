from cmath import log10
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_loss(n,m,k):
    plt.figure(figsize=(8, 7))#窗口大小可以自己设置
    x1,y1 = [],[]
    x2,y2 = [],[]
    x3,y3 = [],[]
    for i in range(2400,2535):
        enc1 = np.load('trace_laser/trace_laser_x/tracex_epoch_{}.npy'.format(i)) #文件返回数组
        enc2 = np.load('trace_laser/trace_laser_y/tracey_epoch_{}.npy'.format(i)) #文件返回数组

        tempy1 = float(enc1.tolist())
        tempy2 = float(enc2.tolist())

        x1.append(tempy1)
        y1.append(tempy2)
    for i in range(128):
        enc1 = np.load('trace_fusion/trace_fusion_x/tracex_epoch_{}.npy'.format(i)) #文件返回数组
        enc2 = np.load('trace_fusion/trace_fusion_y/tracey_epoch_{}.npy'.format(i)) #文件返回数组

        tempy1 = float(enc1.tolist())
        tempy2 = float(enc2.tolist())

        x2.append(tempy1)
        y2.append(tempy2)

    for i in range(204):
        enc1 = np.load('trace_camera/trace_camera_x/tracex_epoch_{}.npy'.format(i)) #文件返回数组
        enc2 = np.load('trace_camera/trace_camera_y/tracey_epoch_{}.npy'.format(i)) #文件返回数组

        tempy1 = float(enc1.tolist())
        tempy2 = float(enc2.tolist())

        x3.append(tempy1)
        y3.append(tempy2)
    # for i in range(610,787):
    #     enc1 = np.load('trace_laser/trace_laser_x2/tracex_epoch_{}.npy'.format(i)) #文件返回数组
    #     enc2 = np.load('trace_laser/trace_laser_y2/tracey_epoch_{}.npy'.format(i)) #文件返回数组

    #     tempy1 = float(enc1.tolist())
    #     tempy2 = float(enc2.tolist())

    #     x1.append(tempy1)
    #     y1.append(tempy2)
    # for i in range(260):
    #     enc1 = np.load('trace_fusion/trace_fusion_x2/tracex_epoch_{}.npy'.format(i)) #文件返回数组
    #     enc2 = np.load('trace_fusion/trace_fusion_y2/tracey_epoch_{}.npy'.format(i)) #文件返回数组

    #     tempy1 = float(enc1.tolist())
    #     tempy2 = float(enc2.tolist())

    #     x2.append(tempy1)
    #     y2.append(tempy2)

    # for i in range(314):
    #     enc1 = np.load('trace_camera/trace_camera_x2/tracex_epoch_{}.npy'.format(i)) #文件返回数组
    #     enc2 = np.load('trace_camera/trace_camera_y2/tracey_epoch_{}.npy'.format(i)) #文件返回数组

    #     tempy1 = float(enc1.tolist())
    #     tempy2 = float(enc2.tolist())

    #     x3.append(tempy1)
    #     y3.append(tempy2)

    
    image = cv2.imread("static_test.png")

    plt.plot(x1, y1, '.-',label='laser')
    plt.plot(x3, y3, '.-',label='camera')
    plt.plot(x2, y2, '.-',label='fusion')



    # plt.grid()#显示网格
    plt_title = 'The Trace Of Robot In Static Environment'
    plt.title(plt_title)#标题名
    plt.xlabel('X')#横坐标名
    plt.ylabel('Y')#纵坐标名
    # plt.ylim(-4, 2)
    plt.legend()#显示曲线信息
    plt.savefig("laser_camera_fusion_trace1.jpg")#当前路径下保存图片名字
    #plt.show()

    trace = cv2.imread("laser_camera_fusion_trace1.jpg")
    image = cv2.resize(image,(trace.shape[1],trace.shape[0]))#统一图片大小
    print(image.shape,trace.shape)
    dst = cv2.addWeighted(image,0.5,trace,0.5,0)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    plot_loss(200,200,200)#文件数量
