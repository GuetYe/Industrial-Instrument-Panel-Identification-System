# 2022/2/18改进：
# 直线拟合
# 历史：表盘中心为图片中点
# 现在：表盘中心通过拟合得到


import cv2
import sys
import numpy as np
import math
from collections import Counter
from numpy.linalg import lstsq
import Connected_Components_With_Stats
#设置参数
connectivity = 4
param = 0
reps = 0.01
aeps = 0.01

def distance(pointer1,pointer2):
    '''
    求两点间距离公式
    :param pointer1:
    :param pointer2:
    :return:
    '''
    p1 = np.array(pointer1)
    p2 = np.array(pointer2)
    p3 = p2 - p1
    p4 = math.hypot(p3[0], p3[1])
    return p4

def Computing_Angle(point1, point2):
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    angle = math.atan((y2 - y1) / (x2 - x1)) * 57.29577
    return angle

def draw_line(img, lines):
    img_copy = img.copy()
    cv2.rectangle(img_copy, (0, 0), (img_copy.shape[0], img_copy.shape[1]), (0, 0, 0), -1)
    # for i in range(0, len(lines)):
    for x1, y1, x2, y2 in lines:
            cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # print(img_copy)
    return img_copy

def fitLine(point):
    '''
    拟合直线并求直线的斜率k和常数项b
    :param point: 直线的两个点
    :return: 斜率k以及常数项b
    '''
    # 直线拟合
    line = cv2.fitLine(point, cv2.DIST_L1, param, reps, aeps)
    # y = k(x-x0) + b
    # y - kx = b - k*x0
    k = (line[1] / line[0])[0]
    x0 = (line[2])[0]
    b = (line[3])[0]
    # 更新参数b 使b = b - k*x0
    b -= k * x0
    return k,b

def SolutionEquations(coefficient, augmented):
    '''
    用最小二乘法求解超定方程组
    :param coefficient: 超定方程组未知量的系数
    :param augmented: 超定方程组未知量的常数项
    :return: 超定方程组的解
    '''
    a = np.mat(coefficient)
    b = np.mat(augmented).T
    x = lstsq(a, b)
    return x

def ScaleProcess(img_scale,img):
    '''
    仪表盘刻度的处理
    :param img_scale: 仪表盘语义信息
    :param img: 原图像
    :return:最大刻度和最小刻度的角度
    '''

    # 用来存放所有的刻度的夹角
    Angle = []
    # 用来存放方程的系数及参数b
    coefficient = []
    augmented = []
    # 图像灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_scale = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)

    cv2.imshow("src_scale", img_scale)

    # 图像自适应二值化
    img = cv2.adaptiveThreshold(img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    img_copy_1 = img.copy()


    img = cv2.add(img,img_scale)

    cv2.imshow("fushion", img)

    Connected_Components_With_Stats.Analysis(img)
    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_8U)

    img_copy = np.zeros([img.shape[0], img.shape[1]],dtype=np.uint8)
    img_fit = img_copy

    for number in range(num_labels):
        if number <= 1:
            continue
        else:
            if centroids[number][1] >= img.shape[1] // 2:
                if centroids[number][0] >= img.shape[0] // 2:
                    point = np.array([[stats[number][0], stats[number][1]],
                                    [stats[number][0] + stats[number][2],
                                     stats[number][1] + stats[number][3]]])
                    # 直线拟合，且获得直线方程斜率k以及常数b
                    k,b = fitLine(point)
                    coefficient.append([1,-k])
                    augmented.append(b)
                    cv2.line(img_copy, (int(stats[number][0]), int(stats[number][1])),
                            (int(stats[number][0] + stats[number][2]), int(stats[number][1] + stats[number][3])),
                             (255, 255, 255))
                else:
                    point = np.array([[stats[number][0], stats[number][1] + stats[number][3]],
                                      [stats[number][0] + stats[number][2],stats[number][1]]])
                    # 直线拟合，且获得直线方程斜率k以及常数b
                    k, b = fitLine(point)
                    # 把所得的k和b放进列表中
                    coefficient.append([1, -k])
                    augmented.append(b)
                    # 直线拟合
                    cv2.line(img_copy, (int(stats[number][0]), int(stats[number][1] + stats[number][3])),
                             (int(stats[number][0] + stats[number][2]), int(stats[number][1])),
                             (255, 255, 255))

            else:
                if centroids[number][0] >= img.shape[0] // 2:
                    point = np.array([[stats[number][0], stats[number][1] + stats[number][3]],
                                      [stats[number][0] + stats[number][2], stats[number][1]]])
                    # 直线拟合，且获得直线方程斜率k以及常数b
                    k, b = fitLine(point)
                    coefficient.append([1, -k])
                    augmented.append(b)
                    # 直线拟合
                    cv2.line(img_copy, (int(stats[number][0]), int(stats[number][1] + stats[number][3])),
                             (int(stats[number][0] + stats[number][2]), int(stats[number][1])),
                             (255, 255, 255))
                else:
                    point = np.array([[stats[number][0], stats[number][1]],
                                      [stats[number][0] + stats[number][2],
                                       stats[number][1] + stats[number][3]]])

                    # 直线拟合，且获得直线方程斜率k以及常数b
                    k, b = fitLine(point)
                    coefficient.append([1, -k])
                    augmented.append(b)

                    cv2.line(img_copy, (int(stats[number][0]), int(stats[number][1])),
                             (int(stats[number][0] + stats[number][2]), int(stats[number][1] + stats[number][3])),
                             (255, 255, 255))

    cv2.imshow("img_copy", img_copy)



    # 最小二乘法求解超定方程组，并把矩阵结果转化为列表

    center_x,center_y = SolutionEquations(coefficient,augmented)[0].tolist()



    # 获得表盘中心坐标
    center = (round(center_x[0]),round(center_y[0]))


    #
    # center = (round(centroids[1][0]),round(centroids[1][1]))

    print("表盘中心坐标为：",center)
    cv2.circle(img_copy_1, center, 2, (255, 255, 255), 3)
    cv2.imshow("src", img_copy_1)

    for number in range(num_labels):
        if number <= 1:
            continue
        else:
            if centroids[number][1] >= center[1] :
                 Angle.append(Computing_Angle((centroids[number][0], centroids[number][1]),center))
                 # cv2.line(img_copy, (int(stats[number][0]), int(stats[number][1])),
                 #            (int(stats[number][0] + stats[number][2]), int(stats[number][1] + stats[number][3])),
                 #             (255, 255, 255))
            cv2.line(img_fit, (int(stats[number][0]), int(stats[number][1])),
                     (int(center[0]), int(center[1])),
                     (255, 255, 255))




    cv2.imshow("img_fit", img_fit)
    print('所有线段角度为', Angle)
    Angle_min, Angle_max = max(Angle), min(Angle)
    return Angle_min, Angle_max,center

def PointerProcess(img_pointer,img,center):
    '''
    指针的处理
    :param img_pointer: 指针语义图像
    :param img: 原图像
    :return: 细化后的指针线段及角度
    '''

    distance_list = []


    img_pointer = cv2.cvtColor(img_pointer, cv2.COLOR_BGR2GRAY)

    img =  cv2.bitwise_not(img_pointer)

    cv2.imshow("fushion_pointer", img)
    # img_ = postprocess(img)

    # 细化
    img = cv2.ximgproc.thinning(img, thinningType=0)
    cv2.imshow("thining", img)


    cv2.imshow("pointer", img)
    # 寻找线段
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=10)
    print("lines=", lines)


    for i in lines:
        for j in i:
            p1= np.array([j[0]+j[2] // 2, center[0]])
            p2 = np.array([j[1]+j[3] //2, center[1]])
            p3 = p2 - p1
            p4 = math.hypot(p3[0], p3[1])
            distance_list.append(p4)
    print(distance_list)
    PositionMin = distance_list.index(max(distance_list))
    pointer = lines[PositionMin]

    img = draw_line(img, pointer)
    # cv2.circle(img, center, 2, (255, 255, 255), 3)
    cv2.imshow("Pointer_line", img)

    Angle = Computing_Angle((pointer[0][0], pointer[0][1]), (pointer[0][2], pointer[0][3]))
    print("角度为",Angle)
    return pointer,Angle

def Identify_scale(pointer_line, pointer_angle, intersection_angle, center,value_min,value_max):
    '''
    根据所得到的信息读数
    :param pointer_line: 指针所在线段
    :param pointer_angle:指针角度
    :param intersection_angle:刻度之间的夹角
    :param center:圆心
    :param value_min:最小量程
    :param value_max:最大量程
    :return: 最终读数
    '''

    center = list(center)   # 表盘中心
    print('表盘中心', center)
    line = pointer_line[0]
    # 指针两端点
    pointer1 = [line[0], line[1]]
    pointer2 = [line[2], line[3]]

    # 两端点到表盘中心的距离
    distance1 = distance(pointer1,center)
    distance2 = distance(pointer2,center)

    # 距离较远的那端来确定指针所在象限
    if distance1 >=distance2:
        line_center = pointer1
    else:
        line_center = pointer2

    print('线段', line)
    print('线段中点',line_center)
    # 确定指针象限后读数,Theta为指针夹角
    if line_center[1] > center[1]:
        if line_center[0] > center[0]:
            Theta = 135 + abs(pointer_angle)
        else:
            Theta = 45 - abs(pointer_angle)
    else:
        if line_center[0] > center[0]:
            Theta = 135 - abs(pointer_angle)
        else:
            Theta = 45 + abs(pointer_angle)

    value_i = value_min + (value_max - value_min) * Theta / intersection_angle
    return value_i

def Identify_scale_test(pointer_line,Angle_min,pointer_angle,intersection_angle,center,value_min,value_max):
    '''
    根据所得到的信息读数
    :param pointer_line: 指针所在线段
    :param pointer_angle:指针角度
    :param intersection_angle:刻度之间的夹角
    :param center:表盘中心
    :param value_min:最小量程
    :param value_max:最大量程
    :return: 最终读数
    '''

    center = list(center)   # 表盘中心
    print('表盘中心', center)
    line = pointer_line[0]
    # 指针两端点
    pointer1 = [line[0], line[1]]
    pointer2 = [line[2], line[3]]

    # 两端点到表盘中心的距离
    distance1 = distance(pointer1,center)
    distance2 = distance(pointer2,center)

    # 距离较远的那端来确定指针所在象限
    if distance1 >=distance2:
        line_center = pointer1
    else:
        line_center = pointer2

    print('线段', line)
    print('线段中点',line_center)
    # 确定指针象限后读数,Theta为指针与最小刻度的夹角
    if line_center[1] > center[1]:
        if line_center[0] > center[0]:
            # 指针在第一象限
            print("指针在第一象限")
            Theta = (90 - abs(Angle_min)) + 180 + abs(pointer_angle)
        else:
            # 指针在第二象限
            print("指针在第二象限")
            Theta = abs(pointer_angle - Angle_min)
    else:
        if line_center[0] > center[0]:
            # 指针在第四象限
            print("指针在第四象限")
            Theta = (180 - abs(Angle_min)) + abs(pointer_angle)
        else:
            # 指针在第三象限
            print("指针在第三象限")
            Theta = (90 - abs(Angle_min)) + abs(pointer_angle)

    # 根据论文中的公式变形，算出最后指针显示得数
    value_i = value_min + Theta / intersection_angle * (value_max - value_min)
    return value_i


def read(img,img_scale,img_pointer,value_min,value_max):
    '''
    仪表盘的处理，包括刻度指针处理以及读数
    :param img: 原图像
    :param img_scale: 图像刻度语义信息
    :param img_pointer: 图像指针语义信息
    :param value_min: 最小量程
    :param value_max: 最大量程
    :return: 最终读数
    '''
    # 角度和中心信息
    Angle_max,Angle_min,center = ScaleProcess(img_scale,img)
    # 最小量程和最大量程之间的夹角
    intersection_angle = 360 - (abs(Angle_max - Angle_min))

    print('刻度线之间的夹角为', intersection_angle)
    # 指针细化出来的线段以及指针角度
    pointer_line,pointer_Angle = PointerProcess(img_pointer, img, center)

    print('指针角度为', pointer_Angle)
    # value = Identify_scale(pointer_line,pointer_Angle,intersection_angle,center,
    #                value_min,value_max)
    # 读数
    value_test = Identify_scale_test(pointer_line,Angle_min,pointer_Angle,intersection_angle,center,
                   value_min,value_max)
    return value_test