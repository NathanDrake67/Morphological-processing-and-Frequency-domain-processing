# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:56:30 2022

Microsoft Windows10 家庭中文版
版本20H2(操作系统内部版本19042.1586)
处理器 lntel(R) Core(TM) i5-8300H CPU @ 2.30GHz2.30 GHz
机带RAM 8.00 GB (7.80 GB可用)
GPU0 lntel(R) UHD Graphics 630
GPU1 NVIDIA GeForce GTX 1050 Ti

@author: 10554
"""

import cv2
import numpy as np
import os
import glob
'''
定义求解器Change函数，用于处理灰度图像，image为输入的灰度图；flag为标志位，num为决定核数
由于采用的是全部为1的结构化元素，故在处理时，该结构化元素与原图像素运算结果取最大值即为膨胀运算
flag=0时，取最大值即为膨胀运算；flag=1时，取最小值即为腐蚀运算
以此为基础，后续的开运算，闭运算，顶帽运算和黑帽运算则均可通过多次重复利用此demo()函数进行运算求解
'''
def demo(image,flag = 0,num = 1):


    h = image.shape[0]                                                #取图像的长h                                    
    w = image.shape[1]                                                #取图像的宽w

    newimage = np.zeros(shape=image.shape, dtype=np.uint8)              #初始化新图像
    for i in range(h):
        for j in range(w):
            a = []
            for k in range(2*num+1):                                  #与结构化元素进行运算，此处即定义了元素边长为2*num+1即为3
                for l in range(2*num+1):
                    if -1<(i-num+k)<h and -1<(j-num+l)<w:             #将这一步运算中，所有窗口内下像素点的值加入上述列表a
                        a.append(image[i-num+k,j-num+l])
            if flag == 0:
                k = max(a)                                            #这里根据调用函数时的目的来决定，若进行膨胀运算，则取列表a中的最大值代替该处的像素值
            else:                               
                k = min(a)                                            #若进行腐蚀运算(flag!=0)，则取列表a中的最大值代替该处的像素值
            newimage[i,j] = k                                         #将处理后的像素点填入新图像
    return newimage       
                   
 

# 读取图片
Image_glob = os.path.join(r'images/',"*.png")     #读取指定文件夹中的所有.png文件
Image_name_list=[]                                #创建空列表Image_name_list
Image_name_list.extend(glob.glob(Image_glob))     #并将其名称添加到列表Image_name_list
#print(Image_name_list[::])                       #可检验一下读取的图片名称及数量
#print(len(Image_name_list))
for i in range(3):                                #设置一个循环，目的是循环读取并滤波处理所有三种图片
    img = cv2.imread(Image_name_list[i])          #读取用来读取图片，返回一个numpy.ndarray类型的多维数组
    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #将图像处理成灰度图

    
    dialate = demo(image)
    erosion = demo(image,1)
    Open = demo(erosion)                           #开运算：先腐蚀后膨胀    
    Close = demo(dialate,1)                        #闭运算：先膨胀后腐蚀
    top_hat =   image - Open                       #顶帽处理：顶帽操作和底帽操作是灰度图像所特有的，其原理是开操作将使峰顶消去，可以用原图减去开操作结果，这样就能得到其消去的部分，而这个过程成为顶帽操作，
    black_hat =  Close - image                     #黑帽处理：底帽操作是用闭操作的结果减去原图就得到被闭操作填充的谷底部分，对应于图像中较暗的部分，也叫黑色底帽。
    
    cv2.imshow("origin",img)
    cv2.imshow('dilation Image',dialate)
    cv2.imshow("erosion Image",erosion)
    cv2.imshow("Black_hat Image",black_hat)
    cv2.imshow("Top_hat Image",top_hat)
    
     #这里设置了判断语句，以分别对三种不同的原图进行分别保存,保存其各自的膨胀、腐蚀、黑帽、顶帽运算结果
    if i == 0:
        cv2.imwrite(os.path.join(r"output_1/","3x3_Dilation_gaussian.png"),dialate)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Erosion_gaussian.png"),erosion)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Black_hat_gaussian.png"),black_hat)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Top_hat_gaussian.png"),top_hat)
    if i == 1:
        cv2.imwrite(os.path.join(r"output_1/","3x3_Dilation_origin.png"),dialate)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Erosion_origin.png"),erosion)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Black_hat_origin.png"),black_hat)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Top_hat_origin.png"),top_hat)
    if i == 2:
        cv2.imwrite(os.path.join(r"output_1/","3x3_Dilation_pepper.png"),dialate)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Erosion_pepper.png"),erosion)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Black_hat_pepper.png"),black_hat)
        cv2.imwrite(os.path.join(r"output_1/","3x3_Top_hat_pepper.png"),top_hat)
    
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 
    
