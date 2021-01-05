#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/05
@Desciption : deepvac yolov5 config, include config, aug, train, val, test, loss
'''


import sys
sys.path.append("../deepvac/")

from deepvac.syszux_config import *


# # # config # # #
config.save_num = 1
config.epoch_num = 200
config.lr = 1e-2
config.lr_step = 70
config.momentum = 0.9
config.lr_factor = 0.1
config.nesterov = False
config.weight_decay = 3e-4
config.device = 'cuda'
config.img_size = 416
config.num_classes = 4
config.strides = [8, 16, 32]
config.model_type = "yolov5l"
config.model_file = "modules/yolo.json"
config.model_path = "output/yolov5l.update.pth"
# config.script_model_dir = f"output/{config.model_type}.torchscript.pt"
# if you want to export model to torchscript, make model.is_training = False first;

# # # loss # # #
config.loss = AttrDict()
config.loss.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
config.loss.box = 0.05
config.loss.cls = 0.5
config.loss.cls_pw = 1
config.loss.obj = 1
config.loss.obj_pw = 1
config.loss.iou_t = 0.2
config.loss.anchor_t = 4
config.loss.fl_gamma = 0

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
config.aug.border = [0, 0] if not config.aug.mosaic else [-config.img_size // 2] * 2

# # # train # # #
config.train = AttrDict()
config.train.shuffle = True
config.train.batch_size = 20
config.train.augment = config.aug
config.train.img_size = config.img_size
config.train.root = "/gemfield/hostpv/PornNewDataset/train"
config.train.annotation = "/gemfield/hostpv/PornNewDataset/train.json"

# # # val # # #
config.val = AttrDict()
config.val.shuffle = False
config.val.batch_size = 15
config.val.augment = config.aug
config.val.img_size = config.img_size
config.val.root = "/gemfield/hostpv/PornNewDataset/val"
config.val.annotation = "/gemfield/hostpv/PornNewDataset/val.json"

# # # test # # #
config.test = AttrDict()
config.test.iou_thres = 0.25
config.test.conf_thres = 0.45
config.test.idx_to_cls = ["Female-Breast", "Female-Gential", "Male-Gential", "Buttock"]
