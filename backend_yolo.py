import os
import numpy as np

import torch

from core.utils_general_yolov3 import set_logging, check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from core.utils_torch_yolov3 import time_synchronized, select_device
from core.utils_yolov3 import letterbox

from backbone.yolov3 import attempt_load
import time

from core.utils import mkdir, to_cpu


@torch.no_grad()
def process_frame_yolo(sys_opt: dict, dataset_loader, det_label_dir: str):
    # opt details: https://github.com/ultralytics/yolov3/blob/master/detect.py
    opt_device = "0"
    opt_img_size = 640
    opt_weights = sys_opt["backbone_yolov3_weight"]
    opt_augment = True

    opt_conf_thres = 0.25
    opt_iou_thres = 0.45
    opt_classes = 0  # filter by class: person
    opt_agnostic_nms = True
    opt_max_det = 1000
    opt_save_conf = True

    # Initialize
    set_logging()
    device = select_device(opt_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    imgsz = opt_img_size

    # Load model
    model = attempt_load(opt_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
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
        n_frames = to_cpu(n_frames).numpy()[0]
        n_digits = len(str(n_frames))

        if batch_idx % 10 == 0:
            print("Test Batch_idx-{}/{}".format(batch_idx, nbatch))

        img0 = to_cpu(key_frame).numpy()[0][:, :, ::-1]  # convert to BGR
        assert img0 is not None, 'Image Not Rendered'

        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
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
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                frame_idx = to_cpu(frame_idx).numpy()[0]
                detection_path = os.path.join(det_label_dir, "{}.txt".format(
                    str(frame_idx + 1).zfill(n_digits)))

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt_save_conf else (cls, *xywh)  # label format
                    with open(detection_path, 'w+') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS)
            print(f'Done. ({t2 - t1:.3f}s)')
