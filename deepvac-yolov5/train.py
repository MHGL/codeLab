#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/05
@Desciption : deepvac yolov5 training, refer to https://github.com/DeepVAC/deepvac/blob/master/deepvac/syszux_deepvac.py
'''


import os
import sys
sys.path.append("/home/liyang/GitHub/codeLab/deepvac")

from modules.model import YoloV5
from deepvac.syszux_log import LOG
from torch.utils.data import DataLoader
from modules.utils_loss import MultiLoss
from data.synthesis import Yolov5MosaicDataset
from deepvac.syszux_deepvac import DeepvacTrain


class DeepvacYolov5Train(DeepvacTrain):
    def __init__(self, config):
        super(DeepvacYolov5Train, self).__init__(config)

    def initNetWithCode(self):
        self.net = YoloV5(self.conf.model_file,
                            self.conf.model_type,
                            self.conf.num_classes,
                            self.conf.strides).to(self.conf.device)
        self.net.is_training = True

    def initTrainLoader(self):
        self.train_dataset = Yolov5MosaicDataset(self.conf.train)
        self.train_loader = DataLoader(dataset=self.train_dataset, 
                                        batch_size=self.conf.train.batch_size,
                                        shuffle=self.conf.train.shuffle, 
                                        num_workers=os.cpu_count(),
                                        collate_fn=Yolov5MosaicDataset.collate_fn)

    def initValLoader(self):
        self.val_dataset = Yolov5MosaicDataset(self.conf.val)
        self.val_loader = DataLoader(dataset=self.val_dataset, 
                                        batch_size=self.conf.val.batch_size,
                                        collate_fn=Yolov5MosaicDataset.collate_fn)

    def initCriterion(self):
        self.criterion = MultiLoss(self.conf)

    def doLoss(self):
        lbox, lobj, lcls, iou = self.criterion(self.output, self.target, self.net)
        self.addScalar('{}/boxLoss'.format(self.phase), lbox, self.epoch)
        self.addScalar('{}/objLoss'.format(self.phase), lobj, self.epoch)
        self.addScalar('{}/clsLoss'.format(self.phase), lcls, self.epoch)
        self.loss = lbox + lobj + lcls
        self.accuracy = iou


if __name__ == '__main__':
    from config import config

    det = DeepvacYolov5Train(config)
    det("request")

