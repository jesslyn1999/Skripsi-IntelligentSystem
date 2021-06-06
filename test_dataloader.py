import time

import torch
from utils.torch_utils import time_synchronized


@torch.no_grad()
def dummy_test_loader(sys_opt: dict, dataset_loader, det_label_dir: str):
    train_dur = 16
    shape = (224, 224)
    channel = 3
    t0 = time.time()
    t_prev = time_synchronized()
    cur_clip = torch.empty(channel, train_dur, *shape)

    for epoch_idx, (frame_ids, n_frames, clips, targets, len_clips, key_frames) in enumerate(dataset_loader):
        batch_size = frame_ids.size(0)
        process_clip = torch.empty(batch_size, channel, train_dur, *shape)

        t1 = time_synchronized()

        for batch_idx, (frame_idx, clip, target, len_clip) in enumerate(zip(frame_ids, clips, targets, len_clips)):
            clip_idx = frame_idx % train_dur
            cur_clip[:, clip_idx:clip_idx + len_clip, :, :] = clip[:, :len_clip, :, :]
            process_clip[batch_idx] = torch.cat((cur_clip[:, clip_idx + 1:, :, :],
                                                 cur_clip[:, 0:clip_idx + 1, :, :]), dim=1)

        t2 = time_synchronized()
        print(f'{epoch_idx}.frames_idx: {frame_ids}. ({t1 - t_prev:.3f}s). part: ({t2 - t1:.3f}s)')
        t_prev = t1

    print(f'Done. ({time.time() - t0:.3f}s)')
