#!/usr/bin/env python
#coding=utf-8

from darkflow.net.build import TFNet
import cv2
from io import BytesIO
import time
import numpy as np
import glob
from PIL import Image, ImageDraw
import os
import requests

 
#选取模型文件和权重训练文件，路径需要自己自定义，tiny-yolo-voc.weights需要导入到bin的文件夹中，这里的阈值可以调整看效果，更低的阈值会提高预测速度,但同时也会造成预测假阳性概率的增加
opt = {"model": "/home/pi/darkflow-master/cfg/tiny-yolo-voc.cfg", "load": "/bin/tiny-yolo-voc.weights", "threshold": 0.15}   

#将设置在TFNnet中应用，并定义为tfnet
tfnet = TFNet(opt)


print('下面开始物体线框绘制。')


#树莓派执行拍照程序
#a=os.system("fswebcam -r 800x600 -S 10 /home/pi/smart/photo/AI-"+time.strftime("%Y_%m_%d")+time.strftime("%H:%M:%S")+".jpg")

#定义初始图片计数1
number = 1

for filename in glob.glob('photo/*.jpg'):
	
	#RGB转换
    img1 = Image.open(filename).convert('RGB')
    
    #将img1进行矩阵化（多维数组），并再次进行颜色转化
    img2 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)

    #应用之前定义的tfnet进行结果预测
    result = tfnet.return_predict(img2)
    print(result)
    
    #重新绘图
    draw = ImageDraw.Draw(img1)
    
    #遍历结果，将框体和结果标出
    for det in result:
		#绘制检测框体
        draw.rectangle([det['topleft']['x'], det['topleft']['y'], 
                        det['bottomright']['x'], det['bottomright']['y']],
                       outline=(255, 0, 0))
                       
        #在框体上方添加预测结果
        draw.text([det['topleft']['x'], det['topleft']['y'] - 13], det['label'], fill=(255, 0, 0))
	
	#存储图片，并将计数添加在名称上
    img1.save('testedphoto/tested %i.jpg' % number)
    #增加图片计数
    number += 1

print("识别后的图片已保存")
	
