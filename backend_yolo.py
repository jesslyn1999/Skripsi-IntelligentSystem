import os
import numpy as np

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from core.utils import logging

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from core.utils import mkdir, to_cpu


@torch.no_grad()
def process_frame_yolo(sys_opt: dict, dataset_loader, det_label_dir: str):
    # opt details: https://github.com/ultralytics/yolov3/blob/master/detect.py
    opt_device = ""
    opt_img_size = 640
    opt_weights = sys_opt["backbone_yolov3_weight"]
    opt_augment = False

    opt_conf_thres = 0.25
    opt_iou_thres = 0.45
    opt_classes = 0  # filter by class: person
    opt_agnostic_nms = False
    opt_max_det = 1000
    opt_save_conf = True

    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    # Initialize
    set_logging()
    device = select_device(opt_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = opt_img_size
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    nbatch = len(dataset_loader)
    mkdir(det_label_dir)

    t0 = time.time()

    for batch_idx, (frame_idx, n_frames, _, target, key_frame) in enumerate(dataset_loader):
        n_frames = to_cpu(n_frames).numpy()
        frame_idx = to_cpu(frame_idx).numpy()

        img0 = to_cpu(key_frame).numpy()[:, :, :, ::-1]  # convert to BGR
        assert img0 is not None, 'Image Not Rendered'

        img = []
        for idx, im in enumerate(img0):
            img.append(letterbox(im, imgsz, stride=stride)[0])
        img = np.asarray(img)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt_augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, opt_classes, opt_agnostic_nms,
                                   max_det=opt_max_det)
        t2 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                n_digits = len(str(n_frames[i]))
                f_idx = frame_idx[i]
                detection_path = os.path.join(det_label_dir, "{}.txt".format(
                    str(f_idx + 1).zfill(n_digits)))

                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (*xyxy, conf) if opt_save_conf else (cls, *xyxy)  # label format
                    with open(detection_path, 'w+') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS)
            # print(f'Done. ({t2 - t1:.3f}s)')

        if batch_idx % 20 == 0:
            logging(f"Progress: [%d/%d]. Done. ({t2 - t1:.3f}s)" % (batch_idx, nbatch))

    print(f'Done. ({time.time() - t0:.3f}s)')
