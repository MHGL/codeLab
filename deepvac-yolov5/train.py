#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/05
@Desciption : deepvac yolov5 training, refer to https://github.com/DeepVAC/deepvac/blob/master/deepvac/syszux_deepvac.py
'''


import os
import sys
sys.path.append("../deepvac")
import torch
import numpy as np

from modules.model import YoloV5
from deepvac.syszux_log import LOG
from modules.utils_loss import MultiLoss
from data.synthesis import Yolov5Dataset
from deepvac.syszux_deepvac import DeepvacTrain


class DeepvacYolov5Train(DeepvacTrain):
    def __init__(self, config):
        super(DeepvacYolov5Train, self).__init__(config)
        self.warmup_iter = self.conf.warmup_epochs * len(self.train_loader)

    def initNetWithCode(self):
        self.net = YoloV5(self.conf.model_file,
                            self.conf.model_type,
                            self.conf.num_classes,
                            self.conf.strides).to(self.conf.device)
        self.net.is_training = True

    def initTrainLoader(self):
        self.train_dataset = Yolov5Dataset(self.conf.train)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.conf.train.batch_size,
                                                        shuffle=self.conf.train.shuffle,
                                                        num_workers=os.cpu_count(),
                                                        collate_fn=Yolov5Dataset.collate_fn)

    def initValLoader(self):
        self.val_dataset = Yolov5Dataset(self.conf.val)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                        batch_size=self.conf.val.batch_size,
                                                        collate_fn=Yolov5Dataset.collate_fn)

    def initSgdOptimizer(self):
        pg0, pg1, pg2 = [], [], []
        for k, v in self.net.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)

        self.optimizer = torch.optim.SGD(pg0, lr=self.conf.lr0, momentum=self.conf.momentum, nesterov=self.conf.nesterov)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.conf.weight_decay})
        self.optimizer.add_param_group({'params': pg2})
        del pg0, pg1, pg2

    def initCriterion(self):
        self.criterion = MultiLoss(self.conf)

    def doLoss(self):
        lbox, lobj, lcls, iou = self.criterion(self.output, self.target, self.net)
        self.addScalar('{}/boxLoss'.format(self.phase), lbox, self.epoch)
        self.addScalar('{}/objLoss'.format(self.phase), lobj, self.epoch)
        self.addScalar('{}/clsLoss'.format(self.phase), lcls, self.epoch)
        self.loss = lbox + lobj + lcls
        self.accuracy = iou

    def preIter(self):
        if self.is_val:
            return
        if self.iter <= self.warmup_iter:
            ni = self.iter
            xi = [0, self.warmup_iter]
            # accumulate
            self.conf.accumulate = max(1, np.interp(ni, xi, [1, self.conf.nbs / self.conf.train.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.conf.warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * self.conf.lr_step(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.conf.warmup_momentum, self.conf.momentum])

    def postIter(self):
        pass

    def process(self):
        self.iter = 0
        epoch_start = self.epoch
        for epoch in range(epoch_start, self.conf.epoch_num):
            self.epoch = epoch
            LOG.logI('Epoch {} started...'.format(self.epoch))
            self.processTrain()
            if self.epoch % 50 == 0:
                self.processVal()
                self.processAccept()


if __name__ == '__main__':
    from config import config

    det = DeepvacYolov5Train(config)
    det("request")

