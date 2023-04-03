# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:55:34 2022
Microsoft Windows10 家庭中文版
版本20H2(操作系统内部版本19042.1586)
处理器 lntel(R) Core(TM) i5-8300H CPU @ 2.30GHz2.30 GHz
机带RAM 8.00 GB (7.80 GB可用)
GPU0 lntel(R) UHD Graphics 630
GPU1 NVIDIA GeForce GTX 1050 Ti

任务二的目的是，将原图及补零后的滤波器算子通过傅里叶变换转换到频域内，再相乘
由于频域内相乘等于时域内卷积，故当对乘积结果进行傅里叶逆变换后，
其效果等同于上一次作业中在时域内直接对图像滤波的效果
经检验对比，本次任务二的处理效果和第一次作业中对应滤波器的处理效果非常接近
验证了时域相乘等效于频域卷积

@author: 10554
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#读取需要处理的两张原图，并转换为灰度图
img = cv2.imread(r'images/gaussian_noise.png',0)
img1 = cv2.imread(r'images/origin_image.png',0)

#读取图片的大小信息，包括图片高度和宽度
size= img.shape
h = img.shape[0]
w = img.shape[1]

#下面是各个滤波器的算子

# 高斯滤波器
x = cv2.getGaussianKernel(3,0)
gaussian = x*x.T

# 拉普拉斯滤波器
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

# Sobel_x滤波器
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# Sobel_y滤波器
sobel_y= np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])

'''注：本例中归一化后的滤波器频谱原点在中心，中间白四周黑即为低通滤波，中间黑四周白即为高通滤波'''
list=[img,img1]
LIST=[gaussian,laplacian,sobel_x,sobel_y]


def demo(src,Filter):
    # 将滤波器补零至图片大小
    iChange = cv2.copyMakeBorder(Filter, int(h/2-2), int(h/2-1), int(w/2-2), int(w/2-1), cv2.BORDER_CONSTANT, value=0)
    '''对滤波器进行傅里叶变换'''
    f1 = np.fft.fft2(iChange)
    fshift1 = np.fft.fftshift(f1)
    #将复数转为浮点数，便于进行傅里叶频谱图显示（滤波器）
    global fimg2
    #对频谱图进行归一化处理，np.log()中+1的意义是去除0值，同时为了避免log后出现负无穷（如+e-7),取log后为0的常值1
    fimg2 = np.log(np.abs(fshift1)+1)
    fimg2 = fimg2/np.max(fimg2)          #后面两句是在进行最大值归一化，保证范围在~255
    fimg2 = fimg2 * 255
    
    
    '''对输入图像进行傅里叶变换处理'''
    #对原图进行傅里叶变化
    f = np.fft.fft2(src)
    # 将频域从左上角移动到中间
    fshift = np.fft.fftshift(f)
    # 将复数转为浮点数,便于进行傅里叶频谱图显示（要处理的图像）
    global fimg
    fimg = np.log(np.abs(fshift))
    
    
    '''原图的频谱与滤波器的频谱在频域内乘积（相当于时域内卷积）'''
    result = fshift * fshift1
    global result1
    #对频谱图进行归一化处理，np.log()中+1的意义是去除0值，同时为了避免log后出现负无穷（如+e-7),取log后为0的常值1
    result1 = np.log(np.abs(result)+1)
    result1 = result1/np.max(result1)          #后面两句是在进行最大值归一化，保证范围在~255
    result1 = result1 * 255    
    
    
    
    # 对乘积结果进行逆傅里叶变换
    if_img1 = np.fft.ifft2(result)
    #此时遵循对称操作的原则，先进行逆变换，再将将频域从左上角移动到中间，否则图像会上下左右颠倒
    ifshift1 = np.fft.ifftshift(if_img1)
    # 将复数转为浮点数,便于进行傅里叶频谱图显示
    global origin_img1
    origin_img1 = np.abs(ifshift1)
    
    '''对未经算子处理的噪声原图进行逆傅里叶变换'''
    ifshift = np.fft.ifftshift(fshift)
    # 将复数转为浮点数进行傅里叶频谱图显示
    ifimg = np.log(np.abs(ifshift))
    if_img = np.fft.ifft2(ifshift)
    # 将复数转为浮点数,便于进行傅里叶频谱图显示
    global origin_img
    origin_img = np.abs(if_img)


# 定义图像显示函数
def show(a,b,c,d,e,f):
    
    plt.rc("font",family="SimHei") #显示中文
    plt.subplot(321), plt.imshow(a, "gray"), plt.title('原图')
    plt.axis('off')
    plt.subplot(322), plt.imshow(b, "gray"), plt.title('原图进行傅里叶变换')
    plt.axis('off')
    plt.subplot(323), plt.imshow(c, "gray"), plt.title('经过傅里叶正逆变换后的原图')
    plt.axis('off')
    plt.subplot(324), plt.imshow(d, "gray"), plt.title('滤波器的傅里叶变换')
    plt.axis('off')
    plt.subplot(325), plt.imshow(np.uint8(e), "gray"), plt.title('滤波器与原图在频域相乘')
    plt.axis('off')
    plt.subplot(326), plt.imshow(f, "gray"), plt.title('频域滤波处理结果')
    plt.axis('off')
    plt.show()
    


'''
设置循环，分别对不同的图像，不同的滤波器进行处理，最终输出保存滤波器的频谱图，
滤波器与原图的频域乘积图，最终对乘积傅里叶逆变换后的效果图(等效时域卷积滤波效果)
'''
for i in range(2):
    for j in range(4):
        demo( list[i] , LIST[j] )
        if i == 0 and j == 0:
            plt.figure("高斯噪声高斯滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)
            cv2.imwrite("./output_2/Fourier_gaussian filter.png",np.uint8(fimg2))
            cv2.imwrite("./output_2/Frequency Domain Product_gaussian filter_gaussian_noise.png",np.uint8(result1))
            
            cv2.imwrite("./output_2/Result_gaussian filter_gaussian_noise.png",np.uint8(origin_img1))
        if i == 0 and j == 1:
            plt.figure("高斯噪声拉普拉斯滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)
            cv2.imwrite("./output_2/Fourier_laplacian filter.png",np.uint8(fimg2))
            cv2.imwrite("./output_2/Frequency Domain Product_laplacian filter_gaussian_noise.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_laplacian filter_gaussian_noise.png",np.uint8(origin_img1))
        if i == 0 and j == 2:
            plt.figure("高斯噪声Sobel_x滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)            
            cv2.imwrite("./output_2/Fourier_sobel_x filter.png",np.uint8(fimg2))
            cv2.imwrite("./output_2/Frequency Domain Product_sobel_x filter_gaussian_noise.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_sobel_x filter_gaussian_noise.png",np.uint8(origin_img1))        
        if i == 0 and j == 3:
            plt.figure("高斯噪声Sobel_y滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)              
            cv2.imwrite("./output_2/Fourier_sobel_y filter.png",np.uint8(fimg2))
            cv2.imwrite("./output_2/Frequency Domain Product_sobel_y filter_gaussian_noise.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_sobel_y filter_gaussian_noise.png",np.uint8(origin_img1))  
        if i == 1 and j == 0:
            plt.figure("原图高斯滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)            
            cv2.imwrite("./output_2/Frequency Domain Product_gaussian filter_origin_image.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_gaussian filter_origin_image.png",np.uint8(origin_img1))
        if i == 1 and j == 1:
            plt.figure("原图拉普拉斯滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)                
            cv2.imwrite("./output_2/Frequency Domain Product_laplacian filter_origin_image.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_laplacian filter_origin_image.png",np.uint8(origin_img1))
        if i == 1 and j == 2:
            plt.figure("原图Sobel_x滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)                
            cv2.imwrite("./output_2/Frequency Domain Product_sobel_x filter_origin_image.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_sobel_x filter_origin_image.png",np.uint8(origin_img1))        
        if i == 1 and j == 3:
            plt.figure("原图Sobel_y滤波器")
            show(img,fimg,origin_img,fimg2,result1,origin_img1)                
            cv2.imwrite("./output_2/Frequency Domain Product_sobel_y filter_origin_image.png",np.uint8(result1))
            cv2.imwrite("./output_2/Result_sobel_y filter_origin_image.png",np.uint8(origin_img1)) 





