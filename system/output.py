from PIL import Image, ImageDraw
from pathlib import Path
from typing import List
import numpy as np
import bbox as bb_util

ALL_ACTION_LABELS = ["falling_down", "chest_pain", "pushing", "touch_pocket",
                     "hit_with_object", "wield_knife", "shoot_with_gun", "support_somebody",
                     "attacked_by_gun", "run"]


def process_image(image_path: str, gt_label_path: str = None, det_label_path: str = None,
                  is_usual: bool = False, is_demo: bool = False):
    img = Image.open(image_path)

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
        img.show()


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


def mkdir(output_folder: str):
    Path(output_folder).mkdir(parents=True, exist_ok=True)


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
    process_image(img_path, gt_file, det_file, is_demo=True)
    pass


if __name__ == '__main__':
    main()
