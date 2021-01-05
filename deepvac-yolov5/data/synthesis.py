#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/04
@Desciption : get annotations from coco annotations and translate to yolov5 custom dataset: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
'''


import os
import cv2
import sys
import torch
import random
sys.path.append("/home/liyang/GitHub/codeLab/deepvac")
import numpy as np

from PIL import Image
from config import config
from torchvision.datasets.vision import VisionDataset
from deepvac.syszux_aug import HSVAug, RandomPerspectiveAug, FlipAug


class CocoDataset(VisionDataset):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.root = deepvac_config.root
        self.annotation = deepvac_config.annotation
        self.transforms = deepvac_config.transforms
        super(CocoDataset, self).__init__(self.root, self.annotation, self.transforms)

        from pycocotools.coco import COCO
        self.coco = COCO(self.annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cats = list(sorted(self.coco.cats.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


class CocoDetectionDataset(CocoDataset):
    def __init__(self, deepvac_config):
        super(CocoDetectionDataset, self).__init__(deepvac_config)

    def _load_image(self, index):
        img_id = self.ids[index]
        image = self.coco.loadImgs(img_id)[0]["file_name"]
        img = cv2.imread(os.path.join(self.root, image), 1)
        assert img is not None, f"image {image} not found!"
        return img

    def _load_anns(self, index):
        img_id = self.ids[index]
        anns_id = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(anns_id)
        anns = [[self.cats.index(i["category_id"]), *i["bbox"]] for i in anns]
        return np.array(anns)

    def _xyxy2xywh(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2 
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2 
        y[:, 2] = x[:, 2] - x[:, 0] 
        y[:, 3] = x[:, 3] - x[:, 1] 
        return y

    def _normalize_hw(self, img, labels):
        nL = len(labels)
        if nL:
            labels[:, 1:5] = self._xyxy2xywh(labels[:, 1:5]) 
            labels[:, [2, 4]] /= img.shape[0] 
            labels[:, [1, 3]] /= img.shape[1] 
        return img, labels
    
    def _convert_tensor(self, img, labels):
        nL = len(labels)
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        return img, labels_out

    def __getitem__(self, index):
        img = self._load_image(index)
        anns = self._load_anns(index)
        img, target = self._normalize_hw(img, anns)
        return self._convert_tensor(img, target)

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch) 
        for i, l in enumerate(label):
            l[:, 0] = i  
        return torch.stack(img, 0), torch.cat(label, 0)


class Yolov5Dataset(CocoDetectionDataset):
    def __init__(self, deepvac_config):
        super(Yolov5Dataset, self).__init__(deepvac_config)
        self.hsv_aug = HSVAug(self.conf.augment)
        self.flip_aug = FlipAug(self.conf.augment)
        self.random_perspective_aug = RandomPerspectiveAug(self.conf.augment)

    def _letterbox(self, img, new_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2] 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup: 
            r = min(r, 1.0)
        ratio = r, r 
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
        if auto: 
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0] 
        dw /= 2 
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
        return img, ratio, (dw, dh)

    def _convert_tensor_with_aug(self, img, labels):
        if self.conf.augment:
            if not self.conf.augment.mosaic:
                img, labels = self.random_perspective_aug(img, labels)
            self.hsv_aug(img)
        img, labels = self._normalize_hw(img, labels)
        if self.conf.augment:
            img, labels = self.flip_aug(img, labels)
        return self._convert_tensor(img, labels)

    def __getitem__(self, index):
        # # # image # # # 
        img = self._load_image(index)
        h0, w0, _ = img.shape
        r = self.conf.img_size / max(h0, w0) 
        if r != 1: 
            interp = cv2.INTER_AREA if r < 1 and not self.conf.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w, _ = img.shape
        hr, wr = h / h0, w / w0
        img, ratio, pad = self._letterbox(img, self.conf.img_size, auto=False, scaleup=self.conf.augment)
        # # # annotation # # #
        labels = self._load_anns(index)
        if labels.size:
            labels[:, 3] = ratio[0] * wr * (labels[:, 1] + labels[:, 3]) + pad[0]
            labels[:, 4] = ratio[1] * hr * (labels[:, 2] + labels[:, 4]) + pad[1]
            labels[:, 1] = ratio[0] * wr * labels[:, 1] + pad[0] 
            labels[:, 2] = ratio[1] * hr * labels[:, 2] + pad[1] 
        return self._convert_tensor_with_aug(img, labels)


class Yolov5MosaicDataset(Yolov5Dataset):
    def __init__(self, deepvac_config):
        super(Yolov5MosaicDataset, self).__init__(deepvac_config)

    def _load_mosaic(self, index):
        labels4 = []
        s = self.conf.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.conf.augment.border]
        indices = [index] + [random.randint(0, len(self.ids)-1) for _ in range(3)]
        for i, index in enumerate(indices):
            # load image
            img = self._load_image(index)
            h0, w0, _ = img.shape
            r = self.conf.img_size / max(h0, w0) 
            if r != 1: 
                interp = cv2.INTER_AREA if r < 1 and not self.conf.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            h, w, _ = img.shape
            hr, wr = h / h0, w / w0
            # load mosaic
            if i == 0: 
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc 
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  
            elif i == 1: 
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: 
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            # load labels
            labels = self._load_anns(index)
            if labels.size: 
                labels[:, 3] = wr * (labels[:, 1] + labels[:, 3]) + padw
                labels[:, 4] = hr * (labels[:, 2] + labels[:, 4]) + padh
                labels[:, 1] = wr * labels[:, 1] + padw
                labels[:, 2] = hr * labels[:, 2] + padh
                labels4.append(labels)
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:]) 
        img4, labels4 = self.random_perspective_aug(img4, labels4)
        return img4, labels4

    def __getitem__(self, index):
        img, labels = self._load_mosaic(index)
        return self._convert_tensor_with_aug(img, labels)


if __name__ == '__main__':
    dataset = Yolov5MosaicDataset(config.train)
    from torch.utils.data import DataLoader


    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=Yolov5Dataset.collate_fn)
    for img, t in loader:
        print(img.shape)
        print(t)