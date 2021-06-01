import random
from PIL import Image
import os
import time
import torch
import numpy as np
from torch.autograd import Variable
from core.utils import convert2cpu, convert2cpu_long, read_truths_args, count_frames
from typing import Tuple, Union, Any
import cv2 as _cv


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if (random.randint(1, 10000) % 2):
        return scale
    return 1. / scale


def random_distort_image(im, dhue, dsat, dexp):
    res = distort_image(im, dhue, dsat, dexp)
    return res


def data_augmentation(clip, shape, jitter, hue, saturation, exposure):
    # Initialize Random Variables
    oh = clip[0].height
    ow = clip[0].width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy

    flip = random.randint(1, 10000) % 2

    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    # Augment
    cropped = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in clip]

    sized = [img.resize(shape) for img in cropped]

    if flip:
        sized = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in sized]

    clip = [random_distort_image(img, dhue, dsat, dexp) for img in sized]

    return clip, flip, dx, dy, sx, sy


def load_data_detection(base_path, imgpath, train, train_dur, sampling_rate, shape, dataset_use='ucf101-24', jitter=0.2,
                        hue=0.1, saturation=1.5, exposure=1.5):
    # clip loading and  data augmentation
    # if dataset_use == 'ucf101-24':
    #     base_path = "/usr/home/sut/datasets/ucf24"
    # else:
    #     base_path = "/usr/home/sut/Tim-Documents/jhmdb/data/jhmdb"

    im_split = imgpath.split('/')
    num_parts = len(im_split)
    im_ind = int(im_split[num_parts - 1][0:5])

    labpath = os.path.join(base_path, "hal-dataset-labels/{}".format(imgpath).replace('.png', '.txt'))

    img_folder = os.path.join(base_path, "hal-dataset/{}".format(os.path.join(im_split[0], im_split[1])))

    if dataset_use == 'ucf101-24':
        max_num = len(os.listdir(img_folder))
    else:
        max_num = len(os.listdir(img_folder)) - 1

    clip = []

    ### We change downsampling rate throughout training as a       ###
    ### temporal augmentation, which brings around 1-2 frame       ###
    ### mAP. During test time it is set to cfg.DATA.SAMPLING_RATE. ###
    d = sampling_rate
    if train:
        d = random.randint(1, 2)

    for i in reversed(range(train_dur)):
        # make it as a loop
        i_temp = im_ind - i * d
        while i_temp < 1:
            i_temp = max_num + i_temp
        while i_temp > max_num:
            i_temp = i_temp - max_num

        if dataset_use == 'ucf101-24':
            path_tmp = os.path.join(img_folder, '{:05d}.jpg'.format(i_temp))
        else:
            path_tmp = os.path.join(img_folder, '{:05d}.png'.format(i_temp))

        tmp_img = Image.open(path_tmp).convert('RGB')

        clip.append(tmp_img)

    if train:  # Apply augmentation
        clip, flip, dx, dy, sx, sy = data_augmentation(clip, shape, jitter, hue, saturation, exposure)
        label = fill_truth_detection(labpath, clip[0].width, clip[0].height, flip, dx, dy, 1. / sx, 1. / sy)
        label = torch.from_numpy(label)

    else:  # No augmentation
        label = torch.zeros(50 * 5)
        try:
            tmp = torch.from_numpy(read_truths_args(labpath, 8.0 / clip[0].width).astype('float32'))
        except Exception:
            tmp = torch.zeros(1, 5)

        tmp = tmp.view(-1)
        tsz = tmp.numel()

        if tsz > 50 * 5:
            label = tmp[0:50 * 5]
        elif tsz > 0:
            label[0:tsz] = tmp

    if train:
        return clip, label
    else:
        return os.path.join(im_split[0], im_split[1], im_split[2]), clip, label


def load_system_data(
        video_path: str, key_frame_idx: int, train_dur: int,
        sampling_rate: int, label_path: str = None
) -> Tuple[int, list[Image.Image], Union[torch.Tensor, Any]]:
    """
    only for testing or system usage, not for training
    key_frame_idx: range 0...len frames of video(exclusive)
    """
    clip = []
    len_video = count_frames(video_path)

    cap = _cv.VideoCapture(video_path)

    for idx in reversed(range(train_dur)):
        tmp_idx = key_frame_idx - idx * sampling_rate
        while tmp_idx < 0:
            tmp_idx = len_video + tmp_idx
        while tmp_idx >= len_video:
            tmp_idx = tmp_idx - len_video

        cap.set(_cv.CAP_PROP_POS_FRAMES, tmp_idx)

        res, frame = cap.read()
        frame = _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        clip.append(frame)

    cap.release()

    key_frame = clip[0].copy()

    """
    Process Label
    """
    label = torch.zeros(50 * 5)
    try:
        tmp = torch.from_numpy(read_truths_args(label_path, 8.0 / clip[0].width).astype('float32'))
    except Exception:
        tmp = torch.zeros(1, 5)

    tmp = tmp.view(-1)
    tsz = tmp.numel()

    if tsz > 50 * 5:
        label = tmp[0:50 * 5]
    elif tsz > 0:
        label[0:tsz] = tmp

    return key_frame_idx, clip, label


# this function works for obtaining new labels after data augumentation
def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes, 5))

    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))

        for i in range(bs.shape[0]):
            cx = (bs[i][1] + bs[i][3]) / (2 * 320)
            cy = (bs[i][2] + bs[i][4]) / (2 * 240)
            imgw = (bs[i][3] - bs[i][1]) / 320
            imgh = (bs[i][4] - bs[i][2]) / 240
            bs[i][0] = bs[i][0] - 1
            bs[i][1] = cx
            bs[i][2] = cy
            bs[i][3] = imgw
            bs[i][4] = imgh

        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2) / 2
            bs[i][2] = (y1 + y2) / 2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label
