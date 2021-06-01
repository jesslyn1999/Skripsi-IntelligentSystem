import torch
import torch.nn as nn
import torch.optim as optim
from backbone.yowo import YOWO
from core.utils import read_data_cfg, str_2_bool
from core.cfg import parse_cfg
import os
from backbone.yowo import get_fine_tuning_parameters
from core.regionloss import RegionLoss
import time
import pathlib
import os.path as _path
from dataset.list_dataset import ListDataset, SystemDataset
from torchvision import transforms
from core.utils import logging, file_lines
from torch.utils.data import DataLoader
from core.optimization import test
from backend_yolo import process_frame_yolo
from system.finalization import process_label_video, video_bbox
from core.utils import Map as _Map

_LOC = _path.realpath(_path.join(os.getcwd(), _path.dirname(__file__)))


def backend_yowo():
    opt = read_data_cfg(_path.join(_LOC, "../config/sys_config.cfg"))

    dataset_use = opt.dataset
    # assert dataset_use == 'ucf101-24' or dataset_use == 'jhmdb-21', 'invalid dataset'

    datacfg = opt.data_cfg
    cfgfile = opt.cfg_file

    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(cfgfile)[0]

    basepath = data_options['base']
    basepath = "/kaggle/input/"
    trainlist = "/kaggle/input/hal-configuration/trainlist.txt"
    testlist = "../input/hal-configuration/demo_testlist.txt"
    backupdir = data_options['backup']

    nsamples = file_lines(trainlist)
    gpus = data_options['gpus']  # e.g. 0,1,2,3 # TODO: add to config e.g. 0,1,2,3
    ngpus = len(gpus.split(','))
    num_workers = int(data_options['num_workers'])

    batch_size = int(net_options['batch'])
    batch_size = 6
    clip_duration = int(net_options['clip_duration'])
    max_batches = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum = float(net_options['momentum'])
    decay = float(net_options['decay'])
    steps = [float(step) for step in net_options['steps'].split(',')]
    scales = [float(scale) for scale in net_options['scales'].split(',')]

    loss_options = parse_cfg(cfgfile)[1]
    anchors = loss_options['anchors'].split(',')
    region_loss = RegionLoss(
        num_classes=opt.n_classes, anchors=[float(i) for i in anchors], batch=batch_size,
        num_anchors=int(loss_options['num'])
    ).cuda()
    region_loss.anchors = [float(i) for i in anchors]
    # region_loss.num_classes    = int(loss_options['classes'])  # TODO: need to changed
    region_loss.num_classes = opt.n_classes
    region_loss.num_anchors = int(loss_options['num_anchors'])
    region_loss.anchor_step = len(region_loss.anchors) // region_loss.num_anchors
    region_loss.object_scale = float(loss_options['object_scale'])
    region_loss.noobject_scale = float(loss_options['noobject_scale'])
    region_loss.class_scale = float(loss_options['class_scale'])
    region_loss.coord_scale = float(loss_options['coord_scale'])
    region_loss.batch = batch_size

    max_epochs = max_batches * batch_size // nsamples + 1

    use_cuda = True
    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    if not os.path.exists(backupdir):
        pathlib.Path(backupdir).mkdir(parents=True, exist_ok=True)

    model_frame = YOWO(opt)

    model_frame = model_frame.cuda()
    # model       = nn.DataParallel(model, device_ids=None) # in multi-gpu case
    model = nn.DataParallel(model_frame, device_ids=['cuda', 'cuda:0'])
    # print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging('Total number of trainable parameters: {}'.format(pytorch_total_params))

    parameters = get_fine_tuning_parameters(model, opt)
    # optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=SOLVER_WEIGHT_DECAY)
    optimizer = optim.SGD(parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)

    best_score = 0  # initialize best score
    best_fscore = 0

    # Load resume path if necessary
    if opt.resume_path:
        print("===================================================================")
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        opt.begin_epoch = checkpoint['epoch'] + 1
        best_fscore = checkpoint['fscore']

        if opt.original:
            model_dict = model.state_dict()
            pretrained_dict = filter_state_dict(model_dict, checkpoint['state_dict'])
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            best_fscore = 0
        else:
            model.load_state_dict(checkpoint['state_dict'])

        if opt.original:
            optimizer_dict = optimizer.state_dict()
            pretrained_dict = filter_optim_state_dict(optimizer_dict, checkpoint['optimizer'])
            optimizer_dict.update(pretrained_dict)
            optimizer.load_state_dict(optimizer_dict)
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

        print("Loaded model fscore: ", checkpoint['fscore'])
        print("===================================================================")

    test_dataset = ListDataset(
        basepath, testlist, dataset_use=dataset_use,
        transform=transforms.Compose([transforms.ToTensor()]),
        train=False, clip_duration=16
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, pin_memory=True
    )

    if opt.evaluate:
        logging('evaluating ...')
        test(0, model, test_loader, region_loss)
    else:
        for epoch in range(opt.begin_epoch, opt.begin_epoch + 1):
            fscore = test(epoch, model, test_loader, region_loss)


def filter_state_dict(ori_state_dict, pre_state_dict, excludes=[]):
    print("Total params in state_dict: {} and {}".format(len(ori_state_dict), len(pre_state_dict)))
    excludes = ["conv_final"]
    output = {}
    for idx, k in enumerate(pre_state_dict.keys()):
        v = pre_state_dict[k]
        if k in ori_state_dict:
            for exclude in excludes:
                if exclude in k:
                    break
            else:
                output[k] = v
    return output


def filter_optim_state_dict(ori_optim, pre_optim):
    keys = ["state",
            "param_groups"]  # value: dict with keys=optim.param_groups["params"], list with length=393 for each
    output = {
        "state": {},
        "param_groups": []
    }
    len_keys_pg = len(pre_optim["param_groups"])  # 393 = len(ori_optim["param_groups"])
    pre_pg = pre_optim["param_groups"]
    pre_state = pre_optim["state"]  # len(pre_state.keys()) = 393

    #     print("size tmp: {}".format(pre_state[pre_pg[392]["params"][0]].keys()))

    for idx, group in enumerate(pre_pg):
        output["param_groups"].append(group)
        for idx2, p in enumerate(group["params"]):
            output["state"][p] = pre_state[p]
            if idx == len_keys_pg - 1:  # exclude conv_final
                del output["state"][p]["momentum_buffer"]

    return output


def generate_dataset_loader(video_path: str, opt_num_workers: int = 0, gt_label_dir: str = None) -> DataLoader:
    system_dataset = SystemDataset(
        video_path, gt_label_dir, shape=(224, 224),
        frame_transform=transforms.Compose([transforms.ToTensor()]),
        clip_dur=16
    )

    return DataLoader(system_dataset, num_workers=opt_num_workers, pin_memory=True)


def process_frame_yowo(sys_opt: dict, test_loader: DataLoader, yowo_det_folder: str):
    """
    frames of the clip must be in RGB and the len match the num of train_frame of the model
    """
    opt_cfg_data = sys_opt["cfg_data"]
    opt_cfg_file = sys_opt["cfg_file"]
    opt_resume_path = sys_opt["resume_path"]
    opt_original = str_2_bool(sys_opt["original"])
    opt_evaluate = str_2_bool(sys_opt["evaluate"])
    opt_begin_epoch = int(sys_opt["begin_epoch"])
    opt_end_epoch = int(sys_opt["end_epoch"])

    sys_data_opt = read_data_cfg(opt_cfg_data)
    sys_cfg_opt = parse_cfg(opt_cfg_file)

    net_opt, region_opt = sys_cfg_opt

    opt_batch_size = int(net_opt["batch_size"])
    opt_momentum = float(net_opt["momentum"])
    opt_decay = float(net_opt["decay"])
    opt_learning_rate = float(net_opt["learning_rate"])

    opt_gpus: str = sys_data_opt["gpus"]  # e.g. 0,1,2,3

    use_cuda = torch.cuda.is_available()
    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt_gpus
        torch.cuda.manual_seed(seed)

    model_frame = YOWO(_Map(sys_opt))

    model_frame = model_frame.cuda()
    model = nn.DataParallel(model_frame, device_ids=["cuda:0"])  # TODO: change for multi gpus

    pytorch_total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logging('Total number of trainable parameters: {}'.format(pytorch_total_params))

    parameters = get_fine_tuning_parameters(model, _Map(sys_opt))
    optimizer = optim.SGD(parameters, lr=opt_learning_rate / opt_batch_size, momentum=opt_momentum, dampening=0,
                          weight_decay=opt_decay * opt_batch_size)

    # Load resume path if necessary
    if opt_resume_path:
        print("===================================================================")
        print('loading checkpoint {}'.format(opt_resume_path))
        checkpoint = torch.load(opt_resume_path)
        opt_begin_epoch = checkpoint['epoch'] + 1

        if opt_original:
            model_dict = model.state_dict()
            pretrained_dict = filter_state_dict(model_dict, checkpoint['state_dict'])
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])

        if opt_original:
            optimizer_dict = optimizer.state_dict()
            pretrained_dict = filter_optim_state_dict(optimizer_dict, checkpoint['optimizer'])
            optimizer_dict.update(pretrained_dict)
            optimizer.load_state_dict(optimizer_dict)
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

        print("Loaded model fscore: ", checkpoint['fscore'])
        print("===================================================================")

    if opt_evaluate:
        logging('evaluating ...')
        test(sys_cfg_opt, 0, model, test_loader, yowo_det_folder)

    else:
        epoch = opt_begin_epoch
        test(sys_cfg_opt, epoch, model, test_loader, yowo_det_folder)
