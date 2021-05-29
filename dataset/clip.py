import random
from PIL import Image
import os
import time
import torch
import numpy as np
from torch.autograd import Variable
from core.utils import convert2cpu, convert2cpu_long, read_truths_args


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


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)  # is it wrong? but well doesnt matter
    w = output.size(3)

    t0 = time.time()
    all_boxes = []

    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(
        5 + num_classes, batch * num_anchors * h * w
    )
    # 5 + num_classes, 1 * 5 * 7 * 7 = 245

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x  # len = 245
    ys = torch.sigmoid(output[1]) + grid_y  # len = 245

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor(
        [0]))  # select only anchor with even index start from 0
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor(
        [1]))  # select only anchors with odd index start from 0
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()  # len = 245
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()  # len = 245
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data  # size: 245, num_classes

    #     print("CLS conf size: {}".format(cls_confs.size()))

    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors  # len=245
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    # order to add in all_boxes: num_anchors, w, h, batch  --> not so important, because nms will have it re-ordered
                    # batch, num_anchors, h, w

                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[
                                    ind] * tmp_conf > conf_thresh:  # TODO: comment if specific
                                    #                                 if c != cls_max_id:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes
