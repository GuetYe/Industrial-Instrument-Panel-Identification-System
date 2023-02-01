import NM_drone
import uwb
import serial
import math
import detect
import time

panel1_coordinate = (4.5,6.0)
panel2_coordinate = (2.5,5.2)
panel3_coordinate = (1.8,1.0)
# 计算方位角函数
def azimuthAngle(point1,point2):
      angle = 0.0
      x1,y1 = point1
      x2,y2 = point2
      dx = x2 - x1
      dy = y2 - y1
      if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1 :
                angle = 0.0
            elif y2 < y1 :
                angle = 3.0 * math.pi / 2.0
      elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
      elif x2 > x1 and y2 < y1 :
            angle = math.pi / 2 + math.atan(-dy / dx)
      elif x2 < x1 and y2 < y1 :
            angle = math.pi + math.atan(dx / dy)
      elif x2 < x1 and y2 > y1 :
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
      return (angle * 180 / math.pi)

if __name__ =="__main__":
    x = []
    y = []
    NM = NM_drone.NM_drone("/dev/ttyUSB1",500000)
    serial = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.05)  # 打开串口
    count = 0   # 计数 用于判断是哪个仪表盘

    info = detect.detect_init()
    RIGHT_rotate_flag = False
    LEFT_rotate_flag = False

    while True:
        coordinate = uwb.get_coordinates(serial)

        if coordinate:

            x.append(coordinate[0])
            y.append(coordinate[1])

            if count == 0:
                angle = azimuthAngle(coordinate, panel1_coordinate) # 计算无人机与目的地的夹角

                if RIGHT_rotate_flag:
                    if (int(angle) - 90) < 0:
                        angle = 359 + (int(angle) - 90)
                    else:
                        angle = int(angle) - 90
                else: angle = int(angle)

                NM.translation(angle=angle)    # 无人机朝目的地前进

                if abs(coordinate[0] - panel1_coordinate[0]) <= 0.1:   #到达目的地，切换至目标识别模式
                    if abs(coordinate[1] - panel1_coordinate[1]) <= 0.1:
                        if RIGHT_rotate_flag == False:
                            NM.right_rotate(angle=90,deg=90)
                            RIGHT_rotate_flag = True
                            time.sleep(1)

                        NM.hover()
                        det_count = detect.run(info=info)


                        if det_count == 0:
                            continue
                        else :
                            NM.left_rotate(angle=90, deg=90)
                            RIGHT_rotate_flag = False
                            count += 1  # 该仪表盘图像采集完毕，切换成下个仪表盘采集


            if count == 1:
                angle = azimuthAngle(coordinate, panel2_coordinate)  # 计算无人机与目的地的夹角

                if LEFT_rotate_flag:
                    if (int(angle) + 90) > 359:
                        angle = int(angle) + 90 - 359
                    else:
                        angle = int(angle) + 90
                else:
                    angle = int(angle)

                NM.translation(angle=angle)  # 无人机朝目的地前进

                if abs(coordinate[0] - panel2_coordinate[0]) <= 0.1:  # 到达目的地，切换至目标识别模式
                    if abs(coordinate[1] - panel2_coordinate[1]) <= 0.1:
                        if LEFT_rotate_flag == False:
                            NM.left_rotate(angle=90, deg=90)
                            time.sleep(1)
                            LEFT_rotate_flag = True


                        NM.translation(angle=angle)  # 旋转后再次判断位置，避免旋转过程中导致的移动
                        if abs(coordinate[0] - panel2_coordinate[0]) <= 0.1:
                            if abs(coordinate[1] - panel2_coordinate[1]) <= 0.1:
                                NM.hover()
                                det_count = detect.run(info=info)

                                if det_count == 0:
                                    continue
                                else:
                                    NM.right_rotate(angle=90, deg=90)
                                    LEFT_rotate_flag = False
                                    count += 1  # 该仪表盘图像采集完毕，切换成下个仪表盘采集


            if count == 2:
                angle = azimuthAngle(coordinate, panel3_coordinate)  # 计算无人机与目的地的夹角
                if LEFT_rotate_flag:
                    if (int(angle) + 90) > 359:
                        angle = int(angle) + 90 - 359
                    else:
                        angle = int(angle) + 90
                else:
                    angle = int(angle)

                NM.translation(angle=angle)  # 无人机朝目的地前进

                if abs(coordinate[0] - panel3_coordinate[0]) <= 0.1:  # 到达目的地，切换至目标识别模式
                    if abs(coordinate[1] - panel3_coordinate[1]) <= 0.1:
                        if LEFT_rotate_flag == False:
                            NM.left_rotate(angle=90, deg=90)
                            time.sleep(1)
                            LEFT_rotate_flag = True

                        NM.hover()
                        det_count = detect.run(info=info)
                        if det_count == 0:
                            continue
                        else:
                            NM.right_rotate(angle=90, deg=90)
                            LEFT_rotate_flag = False
                            count += 1  # 该仪表盘图像采集完毕，切换成下个仪表盘采集

            if count == 3:
                print("coordinate",coordinate,"x0,y0:",[x[0],y[0]])
                angle = azimuthAngle(coordinate, [x[0],y[0]])  # 计算无人机与目的地的夹角

                NM.translation(angle=int(angle))  # 无人机朝目的地前进

                if abs(coordinate[0] - x[0]) <= 0.1:  # 到达目的地，并降落
                    if abs(coordinate[1] - y[0]) <= 0.1:
                        print("飞机降落")
                        NM.land()

            # uwb.draw(x, y)
            print('角度：',angle)