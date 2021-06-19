import PySimpleGUI as sg

import system_main
from core.utils import read_data_cfg
from core.cfg import parse_cfg
import os
from pathlib import Path as _Path
from backend_yolo import process_frame_yolo, test_yolo
from core.optimization import test_yowo
from system.finalization import process_label_video, video_bbox, process_image
from backend_yowo import process_frame_yowo, generate_dataset_loader
import time
import torch
from utils.torch_utils import time_synchronized
import threading
import cv2 as _cv
import numpy as np
from PIL import Image


video_path = "D:\\semester 8\\shootgun-video.mp4"
gt_label_folder = "D:\\semester 8\\TA\\HAL Dataset\\gt_labels\\pushing\\S001C001P002R002A052"
det_label_dir = "D:\\semester 8\\TA\\IntelligentSystem-Result\\thrash-shootgun-video\\det-label"
sys_cfg_path = "D:\\semester 8\\TA\\IntelligentSystem\\config\\system\\sys_config.cfg"
gt_label_folder = ""


video_path_stem = _Path(video_path).stem
yolo_label_folder = os.path.join(det_label_dir, "detection_yolo")
yowo_label_folder = os.path.join(det_label_dir, "detection_yowo")
final_label_folder = os.path.join(det_label_dir, "detection_final")
final_video_path = os.path.join(det_label_dir, "system-{}.avi".format(video_path_stem))
final_vid_yowo_path = os.path.join(det_label_dir, "yowo-{}.avi".format(video_path_stem))


def create_first_layout():
    first_layout = [
        [sg.Text('Action Localization System', size=(40, 1), font=('Any', 18), text_color='#1c86ee',
                 justification='center')],
        [sg.Text('Path to input video'), sg.In(video_path, size=(40, 1), key='input_vid_file'), sg.FileBrowse()],
        [sg.Text('Path to output directory'), sg.In(det_label_dir, size=(40, 1), key='output_dir'), sg.FolderBrowse()],
        [sg.Text('Config'), sg.In(sys_cfg_path, size=(40, 1), key='input_sys_config'), sg.FileBrowse()],
        [sg.Text('Gt_label_folder_path'), sg.In(gt_label_folder, size=(40, 1), key='input_gt_folder'),
         sg.FolderBrowse()],
        [sg.Text(' ' * 8), sg.Checkbox('Write to disk', key='_DISK_')],
        # [sg.OK(), sg.Cancel(), sg.Button('Have detected labels?', pad=((0, 0), 3)), sg.Stretch()],
        [sg.OK(), sg.Cancel(), sg.Stretch()],
    ]
    return first_layout


def create_second_layout():
    second_layout = [
        [sg.Text('Action Localization System Premium', size=(40, 1), font=('Any', 18), text_color='#1c86ee',
                 justification='center')],
        [sg.Text('Path to input video'), sg.In(video_path, size=(40, 1), key='input_vid_file'), sg.FileBrowse()],
        [sg.Text('all_in_one_label_path'), sg.In(det_label_dir, size=(40, 1), key='input_det_folder'),
         sg.FolderBrowse()],
        [sg.Text('YOWO_label_folder'), sg.In(yowo_label_folder, size=(40, 1), key='input_yowo_folder'),
         sg.FolderBrowse()],
        [sg.Text('YOLO_label_folder'), sg.In(yolo_label_folder, size=(40, 1), key='input_yolo_folder'),
         sg.FolderBrowse()],
        [sg.Text('System_label_folder'), sg.In(final_label_folder, size=(40, 1), key='input_final_folder'),
         sg.FolderBrowse()],
        [sg.Text('Gt_label_path'), sg.In(gt_label_folder, size=(40, 1), key='input_gt_folder'), sg.FolderBrowse()],
        [sg.Text('output_video_yowo'), sg.In(final_vid_yowo_path, size=(40, 1), key='output_vid_yowo'),
         sg.FileSaveAs()],
        [sg.Text('output_video_system'), sg.In(final_video_path, size=(40, 1), key='output_vid_sys'), sg.FileSaveAs()],

        [sg.OK(), sg.Cancel(), sg.Button('Back To First Page?', pad=((0, 0), 3)), sg.Stretch()],
    ]
    return second_layout


def first_window():
    return sg.Window('Action Localization System',
                     create_first_layout(),
                     finalize=True,
                     default_element_size=(21, 1),
                     text_justification='right',
                     auto_size_text=False)


def second_window():
    return sg.Window('YOLO Detected Video',
                     create_second_layout(),
                     finalize=True,
                     default_element_size=(21, 1),
                     text_justification='right',
                     auto_size_text=False)


def loading_window(text: str):
    loading_layout = [
        [sg.Text('Loading... {}'.format(text), size=(21, 1), font=('Any', 18), text_color='#1c86ee',
                 justification='center')]]
    return sg.Window('YOLO Detected Video',
                     loading_layout,
                     finalize=True,
                     default_element_size=(21, 1),
                     text_justification='right',
                     auto_size_text=False)


def image_loading():
    global flag
    flag = True
    while flag:
        sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, time_between_frames=100)


DL_START_KEY = '-START DOWNLOAD-'
DL_COUNT_KEY = '-COUNT-'
DL_END_KEY = '-END DOWNLOAD-'


def main():
    sg.ChangeLookAndFeel('LightGreen')
    window_list = [first_window, second_window]

    active_layout = 0
    args = None
    win = window_list[active_layout]()

    while True:
        event, values = win.read()

        if event is None or event == 'Cancel':
            print("Exit .. windows")
            exit()

        if event == "OK":
            args = values
            win.Close()
            break

        if event == "Have detected labels?":
            win.Close()
            active_layout += 1
            win = window_list[active_layout]()

        if event == "Back To First Page?":
            win.Close()
            active_layout -= 1
            win = window_list[active_layout]()

    win.close()

    video_path = args["input_vid_file"]
    det_label_dir = args["output_dir"]
    sys_cfg_path = args["input_sys_config"]
    gt_file_path = args["input_gt_folder"]


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

    """
    dataset_loader = generate_dataset_loader(video_path, opt_num_workers, opt_batch_size)

    win = loading_window("Loading Model")
    t0 = time.time()
    cur_clip = torch.empty(opt_channel, opt_clip_dur, *opt_shape)

    epoch_yowo, model_yowo = process_frame_yowo(sys_opt, yowo_label_folder)
    model_yolo, device, stride = process_frame_yolo(sys_opt, yolo_label_folder)

    t1 = time_synchronized()
    t_prev = t1
    total_frame = -1

    print(f'Load Model: ({t1 - t0:.3f}s)')
    win.Close()

    max_iterate = len(dataset_loader)

    layout = [[sg.Text('Process frames...')],
              [sg.ProgressBar(100, 'h', size=(30, 20), k='-PROGRESS-')]]

    win = sg.Window('Processing frames', layout, finalize=True)
    win.read(timeout=0)
    win.move(win.current_location()[0], win.current_location()[1] - 300)

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
        win['-PROGRESS-'].update(epoch_idx + 1, max_iterate)

    t4 = time_synchronized()
    print(f'done processing: ({t4 - t1:.3f}s). total_frame={total_frame}. fps={total_frame / (t4 - t1):.3f}')
    win.Close()

    win = loading_window("Integrating Video")
    process_label_video(video_path, final_label_folder, yolo_label_folder, yowo_label_folder)
    video_bbox(video_path, final_video_path, det_folder=final_label_folder, gt_folder=gt_label_folder)
    video_bbox(video_path, final_vid_yowo_path, det_folder=yowo_label_folder)
    win.close()
    
    """

    win = loading_window("Integrating Video")
    time.sleep(3)
    win.close()


    """
    TODO: fix gt
    """
    gt_folder = gt_file_path
    det_folder = final_label_folder

    values_spinner = [i for i in range(100)]

    cap = _cv.VideoCapture(video_path)
    width = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(_cv.CAP_PROP_FPS))
    n_frames = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
    n_digits = len(str(n_frames))

    cur_frame = 0
    win_started = False

    thresholds_conf = []
    num_labels = 3
    is_gt_shown = True

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cur_frame += 1

        frame = _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)

        targeted_filename = "{}.txt".format(str(cur_frame).zfill(n_digits))
        gt_label_path = None
        if gt_folder and is_gt_shown:
            gt_label_path = str([path for path in _Path(gt_folder).rglob("*{}.txt".format(cur_frame))
                                 if int(str(path.stem)) == cur_frame][0])

        det_label_path = os.path.join(det_folder, targeted_filename)

        img = process_image(frame, gt_label_path, det_label_path, num_labels=num_labels, is_usual=True)
        img.thumbnail((980, 980), Image.ANTIALIAS)
        img = np.array(img)[:, :, ::-1]

        imgbytes = _cv.imencode('.png', img)[1].tobytes()

        if not win_started:
            win_started = True
            layout = [
                [sg.Text('Playback in PySimpleGUI Window', size=(30, 1))],
                [sg.Image(data=imgbytes, key='_IMAGE_')],
                [[sg.Text("num_labels", size=(20, 1)),
                  sg.Slider(range=(1, 9), orientation='h', resolution=1, default_value=3, size=(15,15), key='num_labels'),
                        sg.T('  ', key='_CONF_OUT_')]],
                [sg.Checkbox('Ground_truths?', default=bool(gt_label_path), key="is_gt")],
                [sg.Exit(), sg.Button('Pause?', pad=((0, 2), 3)), sg.Stretch(), sg.Button('Play?', pad=((0, 2), 3)),
                 sg.Stretch()]
            ]
            win = sg.Window('YOLO Output',
                            default_element_size=(14, 1),
                            text_justification='right',
                            element_justification='c',
                            auto_size_text=False).Layout(layout).Finalize()
            win.Maximize()
            image_elem = win.FindElement('_IMAGE_')
        else:
            image_elem.Update(data=imgbytes)

        event, values = win.Read(timeout=0)
        if event is None or event == 'Exit':
            exit()
        if event == "Pause?":
            event, values = win.Read(timeout=0)
            while event != "Play?":
                event, values = win.Read(timeout=0)
        num_labels = int(values["num_labels"])
        is_gt_shown = values["is_gt"]

    cap.release()
    win.Close()

    win = loading_window("Finish...")
    event, values = win.read()

    if event is None or event == 'Cancel':
        print("Exit .. windows")
        exit()

    win.close()


if __name__ == '__main__':
    main()

"""
Have detectedlabels? 
{'input_vid_folder': 'currently not supported', 'Browse': '',
         'input_vid_file': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\shootgun-video.mp4', 'Browse0': '',
         'output_dir': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label',
         'Browse1': '', 'yolo': 'D:\\semester 8\\TA\\IntelligentSystem\\config\\system\\sys_config.cfg', 'Browse2': '',
         'confidence': 5.0, 'threshold': 3.0, '_WEBCAM_': False, '_DISK_': False}
Back To First Page?
{'input_vid_file': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\shootgun-video.mp4', 'Browse': '',
 'input_det_folder': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label', 'Browse0': '',
 'input_yowo_folder': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label\\detection_yowo',
 'Browse1': '',
 'input_yolo_folder': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label\\detection_yolo',
 'Browse2': '',
 'input_final_folder': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label\\detection_final',
 'Browse3': '', 'input_gt_folder': 'not supported', 'Browse4': '',
 'output_vid_yowo': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label\\yowo-shootgun-video.avi',
 'Save As...': '',
 'output_vid_sys': 'D:\\semester 8\\TA\\IntelligentSystem-Result\\thrashhh-shootgun-video\\det-label\\system-shootgun-video.avi',
 'Save As...5': ''}
"""


def progress_bar():
    # Display a progress meter. Allow user to break out of loop using cancel button
    for i in range(10000):
        if not sg.one_line_progress_meter('My 1-line progress meter',
                                          i + 1, 10000,
                                          'meter key',
                                          'MY MESSAGE1',
                                          'MY MESSAGE 2',
                                          orientation='h',
                                          no_titlebar=True,
                                          grab_anywhere=True,
                                          bar_color=('white', 'red')):
            print('Hit the break')
            break
    for i in range(10000):
        if not sg.one_line_progress_meter('My 1-line progress meter',
                                          i + 1, 10000,
                                          'meter key',
                                          'MY MESSAGE1',
                                          'MY MESSAGE 2',
                                          orientation='v'):
            print('Hit the break')
            break

    layout = [
        [sg.Text('One-Line Progress Meter Demo', font=('Any 18'))],

        [sg.Text('Outer Loop Count', size=(15, 1), justification='r'),
         sg.Input(default_text='100', size=(5, 1), key='CountOuter'),
         sg.Text('Delay'), sg.Input(default_text='10', key='TimeOuter', size=(5, 1)), sg.Text('ms')],

        [sg.Text('Inner Loop Count', size=(15, 1), justification='r'),
         sg.Input(default_text='100', size=(5, 1), key='CountInner'),
         sg.Text('Delay'), sg.Input(default_text='10', key='TimeInner', size=(5, 1)), sg.Text('ms')],

        [sg.Button('Show', pad=((0, 0), 3), bind_return_key=True),
         sg.Text('me the meters!')]
    ]

    window = sg.Window('One-Line Progress Meter Demo', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'Show':
            max_outer = int(values['CountOuter'])
            max_inner = int(values['CountInner'])
            delay_inner = int(values['TimeInner'])
            delay_outer = int(values['TimeOuter'])
            for i in range(max_outer):
                if not sg.one_line_progress_meter('Outer Loop', i + 1, max_outer, 'outer'):
                    break
                time.sleep(delay_outer / 1000)
                for j in range(max_inner):
                    if not sg.one_line_progress_meter('Inner Loop', j + 1, max_inner, 'inner'):
                        break
                    time.sleep(delay_inner / 1000)
    window.close()


def entry_field():
    sg.ChangeLookAndFeel('GreenTan')

    form = sg.FlexForm('Everything bagel', default_element_size=(40, 1))

    column1 = [[sg.Text('Column 1', background_color='#d3dfda', justification='center', size=(10, 1))],
               [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],
               [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],
               [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]]
    layout = [
        [sg.Text('All graphic widgets in one form!', size=(30, 1), font=("Helvetica", 25))],
        [sg.Text('Here is some text.... and a place to enter text')],
        [sg.InputText('This is my text')],
        [sg.Checkbox('My first checkbox!'), sg.Checkbox('My second checkbox!', default=True)],
        [sg.Radio('My first Radio!     ', "RADIO1", default=True), sg.Radio('My second Radio!', "RADIO1")],
        [sg.Multiline(default_text='This is the default Text should you decide not to type anything', size=(35, 3)),
         sg.Multiline(default_text='A second multi-line', size=(35, 3))],
        [sg.InputCombo(('Combobox 1', 'Combobox 2'), size=(20, 3)),
         sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=85)],
        [sg.Listbox(values=('Listbox 1', 'Listbox 2', 'Listbox 3'), size=(30, 3)),
         sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=25),
         sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=75),
         sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=10),
         sg.Column(column1, background_color='#d3dfda')],
        [sg.Text('_' * 80)],
        [sg.Text('Choose A Folder', size=(35, 1))],
        [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
         sg.InputText('Default Folder'), sg.FolderBrowse()],
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('One-Line Progress Meter Demo', layout)

    while True:
        event, values = window.read()
        sg.Popup(event, values)

# if __name__ == "main":
#     demo_one_line_progress_meter()
