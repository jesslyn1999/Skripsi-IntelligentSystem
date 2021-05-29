import random
from dataset.clip import load_data_detection
import torch
from torch.utils.data import Dataset


class listDataset(Dataset):

    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self, base, root, dataset_use='ucf101-24', shape=(224, 224),
                 transform=None, target_transform=None, shuffle=False,  # auto-shuffle from DataLoader
                 train=False, clip_duration=16, sampling_rate=1):

        with open(root, 'r') as file:
            self.lines = file.read().splitlines()

        if shuffle:
            random.shuffle(self.lines)

        # TODO: delete line
        #         self.lines = self.lines[:6]

        self.base_path = base
        self.dataset_use = dataset_use
        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.clip_duration = clip_duration
        self.sampling_rate = sampling_rate

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        imgpath = self.lines[index].rstrip()

        if self.train:  # For Training
            jitter = 0.2
            hue = 0.1
            saturation = 1.5
            exposure = 1.5
            clip, label = load_data_detection(self.base_path, imgpath, self.train, self.clip_duration,
                                              self.sampling_rate, self.shape, self.dataset_use, jitter, hue, saturation,
                                              exposure)

        else:  # For Testing
            frame_idx, clip, label = load_data_detection(self.base_path, imgpath, False, self.clip_duration,
                                                         self.sampling_rate, self.shape, self.dataset_use)
            clip = [img.resize(self.shape) for img in clip]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        # (self.duration, -1) + self.shape = (8, -1, 224, 224)
        clip = torch.cat(clip, 0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return (clip, label, imgpath)
        else:
            return (frame_idx, clip, label, imgpath)
