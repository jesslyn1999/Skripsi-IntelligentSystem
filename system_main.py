from core.utils import read_data_cfg
from core.cfg import parse_cfg
import os
from pathlib import Path as _Path
from backend_yolo import process_frame_yolo, test_yolo
from core.optimization import test_yowo
from system.finalization import process_label_video, video_bbox
from backend_yowo import process_frame_yowo, generate_dataset_loader
import time
import torch
from utils.torch_utils import time_synchronized
import threading


def worker(num):
    """thread worker function"""
    print('Worker:', num)
    return


def main():
    video_path = "D:\\semester 8\\TA\\IntelligentSystem-Result\\shootgun-video.mp4"
    det_label_dir = "D:\\semester 8\\TA\\IntelligentSystem-Result\\thrash-shootgun-video\\det-label"
    sys_cfg_path = "D:\\semester 8\\TA\\IntelligentSystem\\config\\system\\sys_config.cfg"
    # video_path = "D:\\semester 8\\TA\\IntelligentSystem-Result\\S002C002P003R001A045_rgb.avi"
    # det_label_dir = "D:\\semester 8\\TA\\IntelligentSystem-Result\\trash-S002C002P003R001A045_rgb\\det-label"

    sys_opt: dict = read_data_cfg(sys_cfg_path)
    opt_cfg_data = sys_opt["cfg_data"]
    opt_cfg_file = sys_opt["cfg_file"]

    sys_data_opt = read_data_cfg(opt_cfg_data)
    sys_cfg_opt = parse_cfg(opt_cfg_file)
    net_opt, region_opt = sys_cfg_opt

    opt_num_workers = int(sys_data_opt["num_workers"])

    opt_batch_size = int(net_opt["batch_size"])

    opt_clip_dur = 16
    opt_channel = 3
    opt_shape = (224, 224)

    video_path_stem = _Path(video_path).stem
    yolo_label_folder = os.path.join(det_label_dir, "detection_yolo")
    yowo_label_folder = os.path.join(det_label_dir, "detection_yowo")
    final_label_folder = os.path.join(det_label_dir, "detection_final")
    final_video_path = os.path.join(det_label_dir, "system-{}.avi".format(video_path_stem))
    final_vid_yowo_path = os.path.join(det_label_dir, "yowo-{}.avi".format(video_path_stem))

    dataset_loader = generate_dataset_loader(video_path, opt_num_workers, opt_batch_size)

    t0 = time.time()
    cur_clip = torch.empty(opt_channel, opt_clip_dur, *opt_shape)

    epoch_yowo, model_yowo = process_frame_yowo(sys_opt, yowo_label_folder)
    model_yolo, device, stride = process_frame_yolo(sys_opt, yolo_label_folder)

    t1 = time_synchronized()
    t_prev = t1
    total_frame = -1

    print(f'Load Model: ({t1 - t0:.3f}s)')

    for epoch_idx, (frame_ids, n_frames, clips, targets, len_clips, key_frames) in enumerate(dataset_loader):
        batch_size = frame_ids.size(0)
        process_clip = torch.empty(batch_size, opt_channel, opt_clip_dur, *opt_shape)

        if total_frame == -1:
            total_frame = n_frames[0].item()

        t2 = time_synchronized()

        for batch_idx, (frame_idx, clip, target, len_clip) in enumerate(zip(frame_ids, clips, targets, len_clips)):
            clip_idx = frame_idx % opt_clip_dur
            cur_clip[:, clip_idx:clip_idx + len_clip, :, :] = clip[:, :len_clip, :, :]
            process_clip[batch_idx] = torch.cat((cur_clip[:, clip_idx + 1:, :, :],
                                                 cur_clip[:, 0:clip_idx + 1, :, :]), dim=1)

        # test_yowo(sys_cfg_opt, model_yowo, (n_frames, frame_ids, process_clip, targets, key_frames), yowo_label_folder)
        # t2_1 = time_synchronized()
        # test_yolo(model_yolo, (n_frames, frame_ids, key_frames, targets), yolo_label_folder, device, stride)

        yowo_thread = threading.Thread(target=test_yowo, args=(
            sys_cfg_opt, model_yowo, (n_frames, frame_ids, process_clip, targets, key_frames), yowo_label_folder))
        yolo_thread = threading.Thread(target=test_yolo, args=(
            model_yolo, (n_frames, frame_ids, key_frames, targets), yolo_label_folder, device, stride))
        threads = [yowo_thread, yolo_thread]

        for itr in threads:
            itr.start()
        for itr in threads:
            itr.join()

        t3 = time_synchronized()
        # print(f'{epoch_idx}.frames_idx: {frame_ids}. load_data: ({t2 - t_prev:.3f}s). '
        #       f'({t3 - t2:.3f}s): yowo.({t2_1 - t2:.3f}s). yolo.({t3 - t2_1:.3f}s).')
        print(f'{epoch_idx}.frames_idx: {frame_ids}. load_data: ({t2 - t_prev:.3f}s). '
              f'({t3 - t2:.3f}s): yowo & yolo thread')
        t_prev = t2

    t4 = time_synchronized()
    print(f'done processing: ({t4 - t1:.3f}s). total_frame={total_frame}. fps={total_frame / (t4 - t1):.3f}')

    process_label_video(video_path, final_label_folder, yolo_label_folder, yowo_label_folder)
    video_bbox(video_path, final_video_path, det_folder=final_label_folder)
    video_bbox(video_path, final_vid_yowo_path, det_folder=yowo_label_folder)


if __name__ == '__main__':
    main()
    pass
