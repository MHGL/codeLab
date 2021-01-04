#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/04
@Desciption : get annotations from coco annotations and translate to yolov5 custom dataset: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
'''


import os
import cv2
import numpy as np

from tqdm import tqdm
from pycocotools.coco import COCO


def get_anns(annotation):
    res = []
    coco = COCO(annotation)
    ids = list(sorted(coco.imgs.keys()))
    cats = list(sorted(coco.cats.keys()))
    for img_id in ids:
        img = coco.loadImgs(img_id)
        filename = img[0]["file_name"]
        W, H = img[0]["width"], img[0]["height"]
        anns_id = coco.getAnnIds(img_id)
        anns = coco.loadAnns(anns_id)
        anns = [[cats.index(i["category_id"]), *i["bbox"]] for i in anns]
        res.append({filename: annotations})
    return res


def gen_anns(root, annotation, save_dir):
    res = get_anns(annotation)
    for path, _, filenames in os.walk(root):
        for filename in filenames:
            if filename in res.keys():
                continue
            elif filename.replace("_blur", '') in res.keys():
                res[filename] = res[filename.replace("_blur", '')]
            elif filename.replace("_black", '') in res.keys():
                res[filename] = res[filename.replace("_black", '')]
            else:
                res[filename] = [[]]

    coco_annotation = {}
    info = {"year": 2021,
            "version": "v1.0",
            "description": "Pornography Detection",
            "contributor": "__MHGL__",
            "url": "https;//",
            "date_created": "2021/01/04"}

    license = {"id": "001",
                "name": "gemfield",
                "url": "https://"}

    categories = [
                    {"id": 0,
                        "name": "Female-Breast"},
                    {"id": 1,
                        "name": "Female-Genitials"}
                    {"id": 2,
                        "name": "Male-Genitials"}
                    {"id": 3,
                        "name": "Buttock"}
                    ]

    img_id, anns_id = 0, 0
    images, annotations = [], []
    for fn, an in res.items():
        fp = os.path.join(root, fn)
        img = cv2.imread(fp, 1)
        assert img is not None, f"image {fp} not found!"
        h, w, c = img.shape
        img_info = {"img_id": img_id, "file_name": fn, "width": w, "height": h}
        images.append(img_info.copy())

        anns = res[fn]
        if anns[0]:
            anns_info = {"img_id": img_id, "id": anns_id, "category_id": None, "bbox": []}
            annotations.append(anns_info.copy())
            anns_id += 1
        else:
            for ann in anns:
                category_id, *bbox = anns
                anns_info = {"img_id": img_id, "id": anns_id, "category_id": category_id, "bbox": bbox}
                annotations.append(anns_info.copy())
                anns_id += 1
        img_id += 1

    coco_annotation["info"] = info
    coco_annotation["license"] = license
    coco_annotation["categories"] = categories
    coco_annotation["images"] = images
    coco_annotation["annotations"] = annotations

    with open(save_dir, 'w') as f:
        json.dump(coco_annotation, f)


if __name__ == "__main__":
    annotation = "/home/liyang/ai05/PornYoloDataset/annotations.json"
    res = get_anns(annotation)
    print("res: ", res)
