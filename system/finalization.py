from PIL import Image, ImageDraw
from pathlib import Path
from typing import List
import numpy as np
import bbox as bb_util

from core.utils import bbox_iou, mkdir

import cv2 as _cv
import os
from pathlib import Path as _Path

ALL_ACTION_LABELS = ["falling_down", "chest_pain", "pushing", "touch_pocket",
                     "hit_with_object", "wield_knife", "shoot_with_gun", "support_somebody",
                     "attacked_by_gun", "run"]


def filter_bbox(yolo_bboxes, yowo_bboxes, out_label_path: str, width: int, height: int):
    num_gts = len(yolo_bboxes)
    selected_final_bboxes = []

    for i in range(num_gts):
        # x1, y1, x2, y2, localization conf
        box_gt = [yolo_bboxes[i][0], yolo_bboxes[i][1], yolo_bboxes[i][2], yolo_bboxes[i][3], yolo_bboxes[i][4]]
        best_iou = 0
        best_j = -1

        for j in range(len(yowo_bboxes)):
            dboxes = yowo_bboxes[j]
            dboxes[0], dboxes[2] = dboxes[0] / 320 * width, dboxes[2] / 320 * width
            dboxes[1], dboxes[3] = dboxes[1] / 240 * height, dboxes[3] / 240 * height
            iou = bbox_iou(box_gt, dboxes, x1y1x2y2=True)  # iou > 0,5 = TP, iou < 0.5 = FP
            if iou > best_iou:
                best_j = j
                best_iou = iou
            elif iou == best_iou and best_iou != 0:
                print(
                    "OMG UNBELIEVABLE! NEED TO CHANGE CODE with IOU={} : {} and {}".format(iou, yowo_bboxes[best_j][:4],
                                                                                           yowo_bboxes[j][:4]))

        if best_j != -1:
            selected_final_bboxes.append(box_gt[:5] + yowo_bboxes[best_j][5:])

        # if best_iou > iou_thresh:
        #     total_detected += 1
        #     if int(boxes[best_j][6]) == box_gt[6]:
        #         correct_classification += 1
        #
        # if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
        #     correct = correct + 1

    l_bboxes = len(selected_final_bboxes)
    with open(out_label_path, 'w+') as writer:
        for idx, bbox in enumerate(selected_final_bboxes):
            if idx == l_bboxes - 1:
                writer.write("{}".format(" ".join(["{:g}".format(elmt) for elmt in bbox])))
            else:
                writer.write("{}\n".format(" ".join(["{:g}".format(elmt) for elmt in bbox])))


def process_label_video(video_path: str, out_label_folder: str, yolo_label_folder: str, yowo_label_folder: str):
    cap = _cv.VideoCapture(video_path)
    mkdir(out_label_folder)

    width = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
    n_digits = len(str(n_frames))

    cur_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cur_frame += 1

        targeted_filename = "{}.txt".format(str(cur_frame).zfill(n_digits))
        yolo_label_path = os.path.join(yolo_label_folder, targeted_filename)
        yowo_label_path = os.path.join(yowo_label_folder, targeted_filename)
        final_label_path = os.path.join(out_label_folder, targeted_filename)

        yolo_bboxes = read_file_lines(yolo_label_path)
        yowo_bboxes = read_file_lines(yowo_label_path)

        filter_bbox(yolo_bboxes, yowo_bboxes, final_label_path, width, height)

    cap.release()


def process_image(
        img: Image.Image, gt_label_path: str = None, det_label_path: str = None,
        is_usual: bool = True, is_demo: bool = False) -> Image.Image:
    gt_bboxes = []
    if gt_label_path:
        gt_bboxes = read_file_lines(gt_label_path)

    det_bboxes = []
    if det_label_path:
        det_bboxes = read_file_lines(det_label_path)

    if is_usual:
        img = image_bbox(img, gt_bboxes, det_bboxes, 3)

    if is_demo:
        img = image_bbox(img, gt_bboxes, det_bboxes, 10)

    return img


def image_bbox(img: Image.Image, gt_bboxes: List[List[float]], det_bboxes: List[List[float]], num_show_label: int):
    """
    image bbox for system output
    """
    color_list: List[str] = list(bb_util.COLOR_NAME_TO_RGB.keys())
    np_img = np.array(img)

    for idx, gt_box in enumerate(gt_bboxes):
        x1, y1, x2, y2 = gt_box[1:5]
        label = ALL_ACTION_LABELS[int(gt_box[0])]
        bb_util.add(np_img, x1, y1, x2, y2, label, "purple", place_label="bottom")

    for idx, d_box in enumerate(det_bboxes):
        det_labels = np.zeros(len(ALL_ACTION_LABELS))
        itr = iter(d_box[5:])
        for cls_conf, cls_label in zip(itr, itr):
            det_labels[int(cls_label)] = cls_conf * 100

        x1, y1, x2, y2 = d_box[:4]
        desc_sort_idxs = det_labels.argsort()[-1:-1 * (num_show_label + 1):-1]
        label = "\n".join(["{} ... {:g}".format(ALL_ACTION_LABELS[tmp_idx], det_labels[tmp_idx])
                           for tmp_idx in desc_sort_idxs])
        bb_util.add(np_img, x1, y1, x2, y2, label, color_list[desc_sort_idxs[0]],
                    place_label="top")

    return Image.fromarray(np_img)


def video_bbox(video_path: str, out_video_path: str, gt_folder: str = None, det_folder: str = None):
    """
    video bbox for system output
    """
    cap = _cv.VideoCapture(video_path)
    width = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(_cv.CAP_PROP_FPS))
    n_frames = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
    n_digits = len(str(n_frames))

    cv_writer = _cv.VideoWriter(out_video_path, _cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

    cur_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cur_frame += 1

        frame = _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)

        targeted_filename = "{}.txt".format(str(cur_frame).zfill(n_digits))
        gt_label_path = str([path for path in _Path(gt_folder).rglob("*{}.txt".format(cur_frame))
                             if int(str(path)) == cur_frame][0])
        det_label_path = os.path.join(det_folder, targeted_filename)

        img = process_image(frame, gt_label_path, det_label_path, is_usual=True)
        cv_writer.write(img)

    cap.release()


def read_file_lines(file_path):
    if not file_path:
        return []
    f = open(file_path)
    lines = f.read().splitlines()
    lines = [line for line in lines if len(line) != 0]
    unique_lines = list(dict.fromkeys(lines))
    split_lines = [[float(element) for element in line.split(" ")] for line in unique_lines]
    return split_lines


def main():
    # gt_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_yolo"
    # det_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_complete"
    # final_det_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_final"
    # frame_folder = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/frames"
    #
    # manage_det(gt_folder, det_folder, frame_folder, final_det_folder)

    img_path = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/frames/" \
               "complete_chest_pain/S004C001P007R001A045_rgb/00001.png"
    gt_file = ""
    det_file = "/Users/mekari/Desktop/jess/Dataset Thesis/ntu/complete_test_video/detection_results/detections_1_final/" \
               "complete_chest_pain/S004C001P007R001A045_rgb/00001.txt"
    process_image(Image.open(img_path), gt_file, det_file, is_demo=True)
    pass


if __name__ == '__main__':
    main()
