import torch
import torch.nn as nn
from backbone.darknet import Darknet
from backbone.resnext import resnext101
from core.cfam import CFAMBlock


class YOWO(nn.Module):

    def __init__(self, opt):
        super(YOWO, self).__init__()
        self.opt = opt

        ##### 2D Backbone #####
        if opt.backbone_2d == "darknet":
            self.backbone_2d = Darknet(opt.backbone_2d_cfg)
            num_ch_2d = 425  # Number of output channels for backbone_2d
        else:
            raise ValueError("Wrong backbone_2d model is requested. Please select\
                              it from [darknet]")

        if opt.backbone_2d_weights:  # load pretrained weights on COCO dataset
            self.backbone_2d.load_weights(opt.backbone_2d_weights)

            ##### 3D Backbone #####
        if opt.backbone_3d == "resnext101":
            self.backbone_3d = resnext101()
            num_ch_3d = 2048  # Number of output channels for backbone_3d
        else:
            raise ValueError("Wrong backbone_3d model is requested. Please select it from [resnext101, resnet101, \
                             resnet50, resnet18, mobilenet_2x, mobilenetv2_1x, shufflenet_2x, shufflenetv2_2x]")

        if opt.backbone_3d_weights:  # load pretrained weights on Kinetics-600 dataset
            self.backbone_3d = self.backbone_3d.cuda()
            self.backbone_3d = nn.DataParallel(self.backbone_3d,
                                               device_ids=None)  # Because the pretrained backbone models are saved in Dataparalled mode
            pretrained_3d_backbone = torch.load(opt.backbone_3d_weights)
            backbone_3d_dict = self.backbone_3d.state_dict()
            pretrained_3d_backbone_dict = {k: v for k, v in pretrained_3d_backbone['state_dict'].items() if
                                           k in backbone_3d_dict}  # 1. filter out unnecessary keys
            backbone_3d_dict.update(pretrained_3d_backbone_dict)  # 2. overwrite entries in the existing state dict
            self.backbone_3d.load_state_dict(backbone_3d_dict)  # 3. load the new state dict
            self.backbone_3d = self.backbone_3d.module  # remove the dataparallel wrapper

        ##### Attention & Final Conv #####
        self.cfam = CFAMBlock(num_ch_2d + num_ch_3d, 1024)
        self.conv_final = nn.Conv2d(1024, 5 * (opt.n_classes + 4 + 1), kernel_size=1, bias=False)

        self.seen = 0

    def forward(self, input):

        x_3d = input  # Input clip
        x_2d = input[:, :, -1, :, :]

        x_2d = self.backbone_2d(x_2d)
        x_3d = self.backbone_3d(x_3d)

        x_3d = torch.squeeze(x_3d, dim=2)

        x = torch.cat((x_3d, x_2d), dim=1)

        x = self.cfam(x)

        out = self.conv_final(x)

        return out


def get_fine_tuning_parameters(model, opt):
    ft_module_names = ['cfam', 'conv_final']  # Always fine tune 'cfam' and 'conv_final'
    if not opt.freeze_backbone_2d:
        ft_module_names.append('backbone_2d')  # Fine tune complete backbone_3d
    else:
        ft_module_names.append('backbone_2d.models.29')  # Fine tune only layer 29 and 30
        ft_module_names.append('backbone_2d.models.30')  # Fine tune only layer 29 and 30

    if not opt.freeze_backbone_3d:
        ft_module_names.append('backbone_3d')  # Fine tune complete backbone_3d
    else:
        ft_module_names.append('backbone_3d.layer4')  # Fine tune only layer 4

    parameters = []

    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters
