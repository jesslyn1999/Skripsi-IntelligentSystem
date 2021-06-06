from core.utils import read_data_cfg
import os
from pathlib import Path as _Path
from backend_yolo import process_frame_yolo
from system.finalization import process_label_video, video_bbox
from backend_yowo import process_frame_yowo, generate_dataset_loader


def main():
    # video_path = "D:\\semester 8\\TA\\IntelligentSystem-Result\\S002C002P003R001A045_rgb.avi"
    sys_cfg_path = "D:\\semester 8\\TA\\IntelligentSystem\\config\\system\\sys_config.cfg"
    video_path = "D:\\semester 8\\TA\\IntelligentSystem-Result\\S002C002P003R001A045_rgb.avi"
    det_label_dir = "D:\\semester 8\\TA\\IntelligentSystem-Result\\trash-S002C002P003R001A045_rgb\\det-label"
    # det_label_dir = "D:\\semester 8\\TA\\IntelligentSystem-Result\\S002C002P003R001A045_rgb-2\\det-label"

    sys_opt: dict = read_data_cfg(sys_cfg_path)
    opt_cfg_data = sys_opt["cfg_data"]
    sys_data_opt = read_data_cfg(opt_cfg_data)
    opt_num_workers = int(sys_data_opt["num_workers"])
    opt_batch_size = int(sys_data_opt["batch_size"])

    video_path_stem = _Path(video_path).stem
    yolo_label_folder = os.path.join(det_label_dir, "detection_yolo")
    yowo_label_folder = os.path.join(det_label_dir, "detection_yowo")
    final_label_folder = os.path.join(det_label_dir, "detection_final")
    final_video_path = os.path.join(det_label_dir, "system-{}.avi".format(video_path_stem))
    final_vid_yowo_path = os.path.join(det_label_dir, "yowo-{}.avi".format(video_path_stem))

    test_loader = generate_dataset_loader(video_path, opt_num_workers, opt_batch_size)

    # process_frame_yowo(sys_opt, test_loader, yowo_label_folder)
    process_frame_yolo(sys_opt, test_loader, yolo_label_folder)
    # process_label_video(video_path, final_label_folder, yolo_label_folder, yowo_label_folder)
    # video_bbox(video_path, final_video_path, det_folder=final_label_folder)
    # video_bbox(video_path, final_vid_yowo_path, det_folder=yowo_label_folder)


if __name__ == '__main__':
    main()
