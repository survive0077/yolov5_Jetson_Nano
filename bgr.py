import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import os
import sys


from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(im_list,
        weights=ROOT / 'weights/D.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device) # select_device()函数修改terminal第一行输出文字
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if save:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
        video_dir = '1.avi'
        fps = 24
        img_size = (640, 480)
        videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    num = -1
    for im0s in im_list:
        num += 1
        # Run inference
        im = im0s.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        person_number = 0
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=augment, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    if names[int(c)] == 'person':
                        n = (det[:, -1] == c).sum()  # detections per class
                        person_number = f"{n}"
                    else:
                        continue

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if names[c] == 'person':
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    else:
                        continue

            # Stream results
            im0 = annotator.result()
            cv2.putText(im0, 'Number:%s' % person_number, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            # cv2.putText(im0, 'BGR:(%s,%s,%s)' % (num, num, num), (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(im0, 'count:%s' % num, (120, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(im0, 'width:%.2f length:%.2f' % (float(width * real_per_pixel), float(length * real_per_pixel)), (220, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(im0, 'fly height:%.2f' % float((num + 20) * real_per_pixel), (450, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow('new', im0)
            cv2.waitKey(1)  # 1 millisecond
            if save:
                videoWriter.write(im0)

    cv2.destroyAllWindows()
    videoWriter.release()



# 返回两向量之间的角度,°
def cal_angle(x, y):
    angle = 0
    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))
    z_cross = np.cross(x, y)
    z_dot = x.dot(y)
    sin = z_cross / (l_x * l_y)
    cos = z_dot / (l_x * l_y)
    angle_temp = np.arcsin(sin)

    if sin >= 0 and cos >= 0:
        angle = angle_temp

    if sin >= 0 and cos <= 0:
        angle = math.pi - angle_temp

    if sin <= 0 and cos >= 0:
        angle = angle_temp

    if sin <= 0 and cos <= 0:
        angle = -angle_temp - math.pi

    angle_t = angle * 180 / math.pi
    if angle_t >= 0:
        angle_t = 360 - angle_t
    else:
        angle_t = math.fabs(angle_t)
    return angle, angle_t


def draw(img, h_ratio):
    for i in range(l):
        for j in range(w):
            x_now = (0.5 + i) * dx
            y_now = (0.5 + j) * dx
            vector_now = np.array([x_now - fly_coor[0], y_now - fly_coor[1]])
            angle_pi, angle = cal_angle(vector_now, vector)
            L = np.sqrt(vector_now.dot(vector_now))
            s = L / (h_ratio - 1)
            point_now = (int(x_now + s * math.cos(angle_pi)), int(y_now + s * math.sin(angle_pi)))
            short_axes = int(head_r_pixel)
            long_axes = int(head_r_pixel / (1 - 1 / h_ratio))
            if point_now[0] >= 0 and point_now[1] >= 0:
                cv2.ellipse(img, point_now, (long_axes, short_axes), int(angle), 0, 360, black, -1)
            else:
                pass
    cv2.rectangle(img, (0, 0), (length - 1, width - 1), (0, 255, 0), 1)
    cv2.circle(img, fly_coor, 5, (0, 255, 0), -1)

    return img


if __name__ == "__main__":
    head_r_pixel = 1
    dx_real = 1  # 实际行人间距m
    head_r_real = 0.1  # 实际人头半径m
    shoulder_r_real = 0.2  # 实际肩膀半径m
    person_height_real = 1.7  # 实际行人身高m
    fly_height_real = 10  # 实际无人机高度m
    width = 480  # 场景宽度像素点
    length = 640  # 场景长度像素点
    real_per_pixel = head_r_real / head_r_pixel  # 每个像素点对应实际长度
    width_real = int(real_per_pixel * width)
    length_real = int(real_per_pixel * length)
    dx = int(dx_real / real_per_pixel)  # 行人间距像素点
    #head_r = int(head_r_real / real_per_pixel)  # 人头半径像素点
    shoulder_r = int(shoulder_r_real / real_per_pixel)  # 肩膀半径像素点
    fly_height = int(fly_height_real / real_per_pixel)  # 无人机高度像素点
    person_height = int(person_height_real / real_per_pixel)  # 行人高度
    h_ratio = fly_height / person_height
    l = length // dx  # 一行多少人
    w = width // dx  # 一列多少人
    fly_coor = (int(0.5 * length), int(0.5 * width))  # 无人机投影坐标,画面正中心
    vector = np.array([1, 0])  # x轴正方向单位向量
    black = (0, 0, 0)  # BGR,黑色
    white = (255, 255, 255)  # BGR,白色

    im_list = []
    # for i in range(256):
    #     img = np.zeros((width, length, 3), np.uint8)  # 生成一个空灰度图像
    #     img[:] = [i, i, i]                      # BGR,设置背景灰度
    #     im_list.append(draw(img))

    # for i in range(50):
    #     img = np.zeros((width, length, 3), np.uint8)  # 生成一个空灰度图像
    #     img[:] = [0, 0, 0]
    #     cv2.circle(img, fly_coor, i, white, -1)
    #     im_list.append(img)

    # for k in range(0, 256):
    #     img = np.zeros((width, length, 3), np.uint8)  # 生成一个空灰度图像
    #     img[:] = [k, k, k]
    #     for i in range(l):
    #         r = i % 6
    #         for j in range(w):
    #             x_now = int((0.5 + i) * dx)
    #             y_now = int((0.5 + j) * dx)
    #             cv2.circle(img, (x_now, y_now), r, white, -1)
    #     im_list.append(img)

    for i in range(20, 201):
        fly_height = i
        h_ratio = fly_height / person_height
        img = np.zeros((width, length, 3), np.uint8)  # 生成一个空灰度图像
        img[:] = [255, 255, 255]
        im_list.append(draw(img, h_ratio))

    run(im_list)

    # img = np.zeros((width, length, 3), np.uint8)  # 生成一个空灰度图像
    # img[:] = [255, 255, 255]
    # img = draw(img, 200/170)
    # cv2.imshow('new', img)
    # cv2.waitKey(1000)
    sys.exit()

    # cv2.namedWindow("image")
    # cv2.imshow('image', img)
    # cv2.waitKey(10000)  # 显示 10000 ms 即 10s 后消失









