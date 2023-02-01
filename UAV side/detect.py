"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


import NM_drone

import time
import numpy as np

from time import sleep

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized



NM = NM_drone.NM_drone("/dev/ttyUSB1",500000)
# me = tello.Tello()
fbRange = [30000,36000]
pid = [0.1, 0.1, 0.2, 0.2]
pError = 0
UDError = 0

count = 0
load_model = True
@torch.no_grad()




def detect_init(
        dataset = "",
        weights='best.pt',  # model.pt path(s)
        source='0',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):


    global load_model

    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    save_img = False



    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if load_model:
        # Initialize
        set_logging()
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet50', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()


        # Dataloader
    if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            bs = len(dataset)  # batch_size

    else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs



    # Run inference

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    load_model = False
    return [model,device,half,dataset,webcam,save_dir,names,save_img,vid_path,vid_writer,save_txt,save_img, update]


def run(

        info,
        weights='best.pt',  # model.pt path(s)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference

):
        global pError, UDError, count

        model,device,half,dataset,webcam,save_dir,names,save_img,vid_path,vid_writer,save_txt,save_img, update = info

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                im0 = cv2.resize(im0,(1280,720))

                im_copy = im0.copy()
                cv2.imshow("test", im_copy)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        ListC = []
                        ListArea = []
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            cx,cy,area = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            ListC.append([cx,cy])
                            ListArea.append(area)

                    if len(ListArea) != 0:
                        i = ListArea.index(max(ListArea))
                        ImgData = [ListC[i],ListArea[i]]
                    else:
                        ImgData = [[0, 0], 0]

                    print("Center", ImgData[0], "Area", ImgData[1])

                    pError,UDError = track(NM, ImgData,1280, 720, pid, pError,UDError,im_copy)

                    if count >= 3:
                        count = 0
                        cv2.destroyAllWindows()
                        return


                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:

                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        NM.land()
                        break



                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                # if 0xFF == ord('q'):
                #     me.land()
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


        # print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt

def track(NM, info, w, h, pid, pError, udpError,img):
    '''
    追踪函数，用于追踪候选框
    :param NM: 无人机类
    :param info: 候选框信息
    :param w: 候选框的宽
    :param h: 候选框的高
    :param pid: pid信息
    :param pError: 左右误差
    :param udpError: 上下误差
    :param img: 通过图传模块传出来的图片
    :return: 更新两个误差的值
    '''
    area = info[1]  # 候选框的面积
    x,y = info[0]   # 候选框中点坐标
    global count
    fb = 0     # 初始化前进速度
    error = x - w//2    # x轴偏离中点的误差
    uderror = y - h//2  # y轴偏离中点的误差

    # KeyboardControl.getKeyboardInput(me)    # 启动控制界面，主要用于强制无人机降落，键盘上的'Q'降落

    # 使用pid来调整左右飞行幅度，最大速度不超过20cm/s
    lr = pid[0] * error + pid[1] * (error - pError)
    lr = int(np.clip(lr,-25,25))

    # 使用pid来调整上下飞行幅度，最大速度不超过20cm/s
    ud = pid[0] * error + pid[1] * (uderror - udpError)
    ud = int(np.clip(ud, -25, 25))

    if lr > -10 and lr < 0:
        lr = -10
    if lr > 0 and lr < 10:
        lr = 10

    if ud > -10 and ud < 0:
        ud = -10
    if ud > 0 and ud < 10:
        ud = 10


    # 如果候选框面积在所设定范围内，则前进速度置0，且保存图片
    if area > fbRange[0] and area < fbRange[1]:
        NM.hover()
        cv2.imwrite(f'Images/{time.time()}.jpg', img)
        if count < 3:
            count += 1

    # 如果候选框面积大于所设定范围内，后退，反之前进
    if area > fbRange[1]:
       fb = -10

    elif area < fbRange[0] and area != 0:
        fb = 10


    # 如果检测不到候选框，所有操作都置零
    if x == 0 or y ==0:
        fb = 0
        lr = 0
        ud = 0
        error = 0
        uderror = 0
        NM.hover()

    # 通过串口发送指令给无人机
    print("lr:",lr,"ud",ud,"fb",fb)
    NM.send_control(int(lr),int(fb),int(ud),0)
    sleep(0.05)
    return error,uderror

def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    return vars(opt)

if __name__ == "__main__":
    info = detect_init()
    run(info=info)