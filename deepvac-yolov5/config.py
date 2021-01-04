# -*- coding:utf-8 -*-
import sys
sys.path.append("../deepvac/")
from deepvac.syszux_config import *


# # # config # # #
config = AttrDict()
config.save_num = 1
config.epoch_num = 200
config.lr = 1e-2
config.lr_step = 150
config.momentum = 0.9
config.lr_factor = 0.1
config.nesterov = False
config.weight_decay = 3e-4
config.device = 'cuda'
config.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
config.img_size = 416
config.num_classes = 4
config.strides = [8, 16, 32]
config.hyp = "data/hyp.json"
config.model_type = "yolov5l"
config.model_file = "modules/yolo.json"
config.model_path = "output/yolov5l.update.pth"
# config.script_model_dir = f"output/{config.model_type}.torchscript.pt"
# if you want to export model to torchscript, make model.is_training = False first;

# # # aug # # #
config.aug = AttrDict()
config.aug.hgain = 0.015
config.aug.sgain = 0.7
config.aug.vgain = 0.4
config.aug.degrees = 0.0
config.aug.translate = 0.1
config.aug.scale = 0.5
config.aug.shear = 0.0
config.aug.perspective = 0.0
config.aug.flipud = 0.0
config.aug.fliplr = 0.5
config.aug.mosaic = 1.0
config.aug.mixup = 0.0
config.boader = [0, 0] if not config.aug.mosaic else [-config.img_size // 2] * 2

# # # train # # #
config.train = AttrDict()
config.train.shuffle = True
config.train.batch_size = 20
config.train.augment = config.aug
config.train.root = "/gemfield/hostpv/PornCocoDataset/train"
config.train.annotations = "/gemfield/hostpv/PornCocoDataset/train.json"

# # # val # # #
config.val = AttrDict()
config.val.augment = False
config.val.shuffle = False
config.val.batch_size = 15
config.val.root = "/gemfield/hostpv/PornCocoDataset/val"
config.val.annotations = "/gemfield/hostpv/PornCocoDataset/val.json"

# # # test # # #
config.test = AttrDict()
config.test.iou_thres = 0.25
config.test.conf_thres = 0.45
config.test.idx_to_cls = ["Female-Breast", "Female-Gential", "Male-Gential", "Buttock"]