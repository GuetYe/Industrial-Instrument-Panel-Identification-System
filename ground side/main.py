import cv2
import sys
import numpy as np
import math
import Unet.segmentation as segmentation
import yolov7.detect as detect
import PaddleOCR.predict_system as OCR
from collections import Counter
from PIL import Image
import ReadData
import os

# 设置参数
dp = 2
min_dist = 20
param1 = 200
param2 = 200
min_radius = 20
max_radius = 200


path = "det/"


def CutImage(img, values):
    for i in values[0, :]:
        print(int(i[0]), int(i[1]), int(i[2]))
        x1, x2, y1, y2 = int(i[0] - i[2]) - 1, int(i[0] + i[2]) + 1, int(i[1] - i[2]) - 1, int(i[1] + i[2]) + 1
        # 裁剪图像
        img = img[y1:y2, x1:x2, :]
    return img



if __name__ == '__main__':
    src = input('请输入要识别的仪表盘：')
    value_min = float(input("请输入量程最小值："))
    value_max = float(input("请输入量程最大值："))

    image = cv2.imread(src)
    # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    print(image.shape)
    cv2.imshow('Origin', image)



    # 目标检测
    image = detect.detection(src)
    cv2.imshow('detection', image)
    # cv2.imwrite(os.path.join(path, 'test.jpg'), image)

    # OCR
    # txts = OCR.predict("det/test.jpg")
    # value_list = []

    #筛选出量程最小值和最大值
    # for value in txts:
    #     value_list.append(float(value))
    # value_max = max(value_list)
    # value_min = min(value_list)
    #
    # # 如果识别的最小值为正数，记为0
    # if value_min >= 0:
    #     value_min = 0
    print("最小刻度为",value_min,"最大刻度为",value_max)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #
    # # 高斯滤波
    # gray = cv2.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)
    #
    #
    #
    # # 检测圆形
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
    #                            param1=param1, param2=param2, minRadius=min_radius,
    #                            maxRadius=max_radius)
    #
    # image = CutImage(image,circles)

    cv2.imshow("cut", image)
    # cv2转PIL，为了后续能用语义分割模型
    image_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 语义分割
    img_pointer,img_scale,img_seg = segmentation.predict(image_PIL)
    cv2.imshow("scale",img_scale)
    cv2.imshow("pointer", img_pointer)
    cv2.imshow("blend", img_seg)
    # 读数
    scale = ReadData.read(image,img_scale,img_pointer,value_min,value_max)

    print('检测结果为：', scale)

    cv2.waitKey(0)
    cv2.destroyAllWindows()