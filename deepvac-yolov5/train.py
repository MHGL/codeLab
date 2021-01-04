# -*- coding:utf-8 -*-
import json
import torch

from modules.model import YoloV5
from deepvac.syszux_log import LOG
from data.datasets import create_dataloader
from modules.utils_loss import compute_loss
from deepvac.syszux_deepvac import DeepvacTrain


class DeepvacYolov5Detect(DeepvacTrain):
    def __init__(self, config):
        super(DeepvacYolov5Detect, self).__init__(config)

    def initNetWithCode(self):
        self.net = YoloV5(self.conf.model_file,
                            self.conf.model_type,
                            self.conf.num_classes,
                            self.conf.strides).to(self.conf.device)
        with open(self.conf.hyp, 'r') as f:
            hyp = json.load(f)
        self.net.hyp = hyp
        self.net.gr = self.conf.gr
        self.net.is_training = True

    def initTrainLoader(self):
        self.train_loader, self.train_dataset = create_dataloader(self.conf.train.root,
                                                                    self.conf.train.annotations,
                                                                    self.conf.hyp,
                                                                    self.conf.img_size,
                                                                    self.conf.train.batch_size,
                                                                    self.conf.train.augment)

    def initValLoader(self):
        self.val_loader, self.val_dataset = create_dataloader(self.conf.val.root,
                                                                self.conf.val.annotations,
                                                                self.conf.hyp,
                                                                self.conf.img_size,
                                                                self.conf.val.batch_size,
                                                                self.conf.val.augment)

    def initCriterion(self):
        self.criterion = compute_loss

    def doLoss(self):
        lbox, lobj, lcls, iou = self.criterion(self.output, self.target, self.net)
        self.addScalar('{}/boxLoss'.format(self.phase), lbox, self.epoch)
        self.addScalar('{}/objLoss'.format(self.phase), lobj, self.epoch)
        self.addScalar('{}/clsLoss'.format(self.phase), lcls, self.epoch)
        self.loss = 0.2 * lbox + 0.5 * lobj + 0.3 * lcls
        self.accuracy = iou


if __name__ == '__main__':
    from config import config

    det = DeepvacYolov5Detect(config)
    det("request")

