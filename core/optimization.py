import torch
import numpy as np
import os
from core.utils import bbox_iou, logging, get_region_boxes, nms, mkdir, to_cpu
from typing import List
from torch.utils.data import DataLoader
from typing import Tuple


def adjust_learning_rate(optimizer, epoch):
    #     lr = learning_rate
    #     for i in range(len(steps)):
    #         scale = scales[i] if i < len(scales) else 1
    #         if batch >= steps[i]:
    #             lr = lr * scale
    #             if batch == steps[i]:
    #                 break
    #         else:
    #             break
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr/batch_size
    #     return lr

    TRAIN_LEARNING_RATE = 1e-4
    SOLVER_LR_DECAY_RATE = 0.5
    SOLVER_STEPS = [3, 4, 5, 6]

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = TRAIN_LEARNING_RATE * (SOLVER_LR_DECAY_RATE ** (sum(epoch >= np.array(SOLVER_STEPS))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new


@torch.no_grad()
def test(sys_cgf_opt: List[dict], epoch, model, test_loader: DataLoader, det_label_dir: str):
    def truths_length(truths):
        for idx in range(50):
            if truths[idx][1] == 0:
                return idx

    nms_thresh = 0.4
    iou_thresh = 0.5
    eps = 1e-5

    net_opt, region_opt = sys_cgf_opt

    opt_num_classes = int(region_opt["num_classes"])
    opt_anchors = [float(itr.strip()) for itr in region_opt["anchors"].split(",")]
    opt_num_anchors = int(region_opt["num_anchors"])

    conf_thresh_valid = 0.005
    total = 0.0
    proposals = 0.0
    correct = 0.0
    fscore = 0.0

    correct_classification = 0.0
    total_detected = 0.0

    nbatch = len(test_loader)
    mkdir(det_label_dir)

    logging('validation at epoch %d' % epoch)
    model.eval()

    logging("Start Progressing...")

    for batch_idx, (frame_idx, n_frames, data, target, key_frame) in enumerate(test_loader):
        n_frames = to_cpu(n_frames).numpy()
        frame_idx = to_cpu(frame_idx).numpy()

        data = data.cuda()

        with torch.no_grad():
            output = model(data).data
            all_boxes = get_region_boxes(output, conf_thresh_valid, opt_num_classes,
                                         opt_anchors, opt_num_anchors, 0, 1)

            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)

                height, width = key_frame[i].size()[:2]

                n_digits = len(str(n_frames[i]))

                f_idx = frame_idx[i]
                detection_path = os.path.join(det_label_dir, "{}.txt".format(
                    str(f_idx + 1).zfill(n_digits)))

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = max(round(float(box[0] - box[2] / 2.0) * width), 0)
                        y1 = max(round(float(box[1] - box[3] / 2.0) * height), 0)
                        x2 = min(round(float(box[0] + box[2] / 2.0) * width), width - 1)
                        y2 = min(round(float(box[1] + box[3] / 2.0) * height), height - 1)

                        # det_conf = float(box[4])
                        # for j in range((len(box) - 5) // 2):
                        #     cls_conf = float(box[5 + 2 * j].item())
                        #     prob = det_conf * cls_conf

                        line = np.append([x1, y1, x2, y2], box[4:])
                        f_detect.write(("{:g} " * line.shape[0]).rstrip().format(*line) + '\n')

                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)

                total = total + num_gts
                pred_list = []

                for bbox in range(len(boxes)):
                    if boxes[bbox][4] > 0.25:
                        proposals = proposals + 1
                        pred_list.append(bbox)

                for gt_idx in range(num_gts):
                    box_gt = [truths[gt_idx][1], truths[gt_idx][2], truths[gt_idx][3], truths[i][4],
                              1.0, 1.0, truths[gt_idx][i]]
                    best_iou = 0
                    best_j = -1

                    for j in pred_list:
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)  # iou > 0,5 = TP, iou < 0.5 = FP
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > iou_thresh:
                        total_detected += 1
                        if int(boxes[best_j][6]) == box_gt[6]:
                            correct_classification += 1

                    if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                        correct = correct + 1

            precision = 1.0 * correct / (proposals + eps)
            recall = 1.0 * correct / (total + eps)
            fscore = 2.0 * precision * recall / (precision + recall + eps)

            if batch_idx % 20 == 0:
                logging("Progress: [%d/%d]" % (batch_idx, nbatch))
            # logging("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))

    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    locolization_recall = 1.0 * total_detected / (total + eps)

    print("Classification accuracy: %.3f" % classification_accuracy)
    print("Locolization recall: %.3f" % locolization_recall)

    return fscore


@torch.no_grad()
def test_yowo(sys_cfg_opt: List[dict], model, tuple_itr, det_label_dir: str):
    def truths_length(truths):
        for idx in range(50):
            if truths[idx][1] == 0:
                return idx

    nms_thresh = 0.4
    conf_thresh_valid = 0.005

    net_opt, region_opt = sys_cfg_opt

    opt_num_classes = int(region_opt["num_classes"])
    opt_anchors = [float(itr.strip()) for itr in region_opt["anchors"].split(",")]
    opt_num_anchors = int(region_opt["num_anchors"])

    n_frame, frame_idx, clip, target, key_frames = tuple_itr

    data = clip.cuda()

    with torch.no_grad():
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh_valid, opt_num_classes,
                                     opt_anchors, opt_num_anchors, 0, 1)

        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)

            height, width = key_frames[i].shape[:2]

            n_digits = len(str(n_frame[i].item()))
            f_idx = frame_idx[i].item()

            detection_path = os.path.join(det_label_dir, "{}.txt".format(
                str(f_idx + 1).zfill(n_digits)))

            with open(detection_path, 'w+') as f_detect:
                for box in boxes:
                    x1 = max(round(float(box[0] - box[2] / 2.0) * width), 0)
                    y1 = max(round(float(box[1] - box[3] / 2.0) * height), 0)
                    x2 = min(round(float(box[0] + box[2] / 2.0) * width), width - 1)
                    y2 = min(round(float(box[1] + box[3] / 2.0) * height), height - 1)

                    line = np.append([x1, y1, x2, y2], box[4:])
                    f_detect.write(("{:g} " * line.shape[0]).rstrip().format(*line) + '\n')
