import os
import numpy as np

import torch
import torchvision.transforms as transforms

from core.utils_transform_yolo import DEFAULT_TRANSFORMS, Resize
from core.utils_box_yolo import non_max_suppression, rescale_boxes, to_cpu

from backbone import yolov3 as models

from pathlib import Path
from core.utils import mkdir


def detect_image(model, image, transform: transforms.Compose, img_size=416, conf_thres=0.5, nms_thres=0.5):
    # Configure input
    input_img = transform((image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    detections = model(input_img)
    detections = non_max_suppression(detections, conf_thres, nms_thres)
    detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections


def process_frame_yolo(dataset_loader, det_label_dir: str, img_size_cvt=416):
    # TODO: Load the YOLO model
    model = models.load_model(
        "<PATH_TO_YOUR_CONFIG_FOLDER>/yolov3.cfg",
        "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights")

    transform = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size_cvt)])

    nbatch = len(dataset_loader)

    model.eval()  # Set model to evaluation mode

    for batch_idx, (frame_idx, n_frames, data, target) in enumerate(dataset_loader):
        n_digits = len(str(n_frames))

        if batch_idx % 10 == 0:
            print("Test Batch_idx-{}/{}".format(batch_idx, nbatch))

        data = data.cuda()

        with torch.no_grad():
            detections = detect_image(model, data[0], transform)
        boxes = to_cpu(detections).numpy()

        detection_path = os.path.join(det_label_dir, str(frame_idx + 1).zfill(n_digits))
        detection_dir_path = Path(detection_path).parent

        mkdir(detection_dir_path)

        with open(detection_path, 'w+') as f_detect:
            for box in boxes:
                if box[4] == 0:  # person
                    det_conf = float(box[4])
                    f_detect.write(str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + " " +
                                   str(det_conf))
