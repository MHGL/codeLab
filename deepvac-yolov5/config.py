#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/05
@Desciption : deepvac yolov5 config, include config, aug, train, val, test, loss
'''


import sys
sys.path.append("../deepvac/")
import math

from deepvac.syszux_config import *


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

# # # config # # #
config.save_num = 1
config.epoch_num = 1500
config.disable_git = False
config.nbs = 64  # nominal batch size
config.accumulate = max(round(config.nbs / config.train.batch_size), 1)
# optimizer
config.lr0 = 0.01
config.lrf = 0.2
config.lr_step = one_cycle(1, config.lrf, config.epoch_num)
config.momentum = 0.937
config.nesterov = True
config.weight_decay = 5e-4  # hyp['weight_decay'] *= total_batch_size * accumulate / nbs
# warmup
config.warmup_epochs = 3.0
config.warmup_bias_lr = 0.1
config.warmup_momentum = 0.8
# model
config.device = 'cuda'
config.img_size = 416
config.num_classes = 4
config.strides = [8, 16, 32]
config.model_type = "yolov5l"
config.model_file = "modules/yolo.json"
config.model_path = "output/yolov5l_nc4.pkl"
config.script_model_dir = f"output/{config.model_type}.torchscript.pt"
# if you want to export model to torchscript, make model.is_training = False first;

# # # loss # # #
config.loss = AttrDict()
config.loss.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
config.loss.box = 0.05
config.loss.cls = 0.00125  # hyp['cls'] *= nc / 80. origin=0.025
config.loss.cls_pw = 1
config.loss.obj = 0.4225  # hyp['obj'] *= imgsz ** 2 / 640. ** 2 * 3. / nl
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
config.aug.flipud = 0.0
config.aug.fliplr = 0.5
config.aug.mosaic = 1.0
config.aug.mixup = 0.0
config.aug.perspective = 0.0

# # # train # # #
config.train = AttrDict()
config.train.shuffle = True
config.train.batch_size = 3
config.train.augment = config.aug
config.train.img_size = config.img_size
config.train.root = "/home/liyang/ai05/PornNewDataset/test"
config.train.annotation = "/home/liyang/ai05/PornNewDataset/test.json"

# # # val # # #
config.val = AttrDict()
config.val.shuffle = False
config.val.batch_size = 9
config.val.augment = config.aug
config.val.img_size = config.img_size
config.val.root = "/home/liyang/ai05/PornNewDataset/test"
config.val.annotation = "/home/liyang/ai05/PornNewDataset/test.json"

# # # test # # #
config.test = AttrDict()
config.test.iou_thres = 0.25
config.test.conf_thres = 0.45
config.test.idx_to_cls = ["Female-Breast", "Female-Gential", "Male-Gential", "Buttock"]
