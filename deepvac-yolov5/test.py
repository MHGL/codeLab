import sys
sys.path.append("../deepvac")
import cv2
import torch
import numpy as np

from torchvision import ops
from modules.model import YoloV5
from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import Deepvac


class Yolov5Detection(Deepvac):
    def __init__(self, conf):
        conf.disable_git = True
        super(Yolov5Detection, self).__init__(conf)

    def initNetWithCode(self):
        self.net = YoloV5(self.conf.model_file,
                            self.conf.model_type,
                            self.conf.num_classes,
                            self.conf.strides).to(self.conf.device)
        self.net.is_training = False

    def _image_process(self, image):
        # letterbox
        h, w, c = image.shape
        r = min(self.conf.img_size / h, self.conf.img_size / w)
        w_new, h_new = int(round(w * r)), int(round(h * r))
        dw, dh = (self.conf.img_size - w_new), (self.conf.img_size - h_new)
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        dw /= 2; dh /= 2
        if h != h_new and w != w_new:
            img = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        else:
            img = image
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # image process
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0
        img = img.unsqueeze(0)
        return img.to(self.device)

    def _post_process(self, preds):
        '''
            preds: [x1, y1, w, h, confi, one-hot * num_classes]
            return: [x1, y1, x2, y2, confi, cls)
        '''
        pred = preds[preds[..., 4] > self.conf.test.conf_thres]  # filter with classifier confidence
        if not pred.shape[0]:
            return torch.zeros((0, 6))
        # Compute conf
        pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, 0] = pred[:, 0] - pred[:, 2] / 2.0
        pred[:, 1] = pred[:, 1] - pred[:, 3] / 2.0
        pred[:, 2] += pred[:, 0]
        pred[:, 3] += pred[:, 1]
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, idx = pred[:, 5:].max(dim=1, keepdim=True)
        pred = torch.cat((pred[:, :4], conf, idx), dim=1)[conf.view(-1) > self.conf.test.conf_thres]  # filter with bbox confidence
        if not pred.shape[0]:
            return torch.zeros((0, 6))
        # nms on per class
        max_side = 4096
        class_offset = pred[:, 5:6] * max_side
        boxes, scores = pred[:, :4] + class_offset, pred[:, 4]
        idxs = ops.nms(boxes, scores, self.conf.test.iou_thres)
        pred = torch.stack([pred[i] for i in idxs], dim=0)
        return pred

    def _plot_rectangle(self, img, pred, file_path):
        save_dir = "output/detect"
        file_name = file_path.split('/')[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n, c, h, w = img.shape
        image = cv2.imread(file_path)
        h0, w0, c0 = image.shape

        if not len(pred):
            cv2.imwrite(os.path.join(save_dir, file_name), image)
            return

        for det in pred:
            coord = det[:4]
            gain = min(h / h0, w / w0)
            pad = (w - w0 * gain) / 2, (h - h0 * gain) / 2
            print(coord)
            coord[[0, 2]] -= pad[0]
            coord[[1, 3]] -= pad[1]
            coord /= gain
            coord = [int(x.item()) for x in coord]
            if (max(coord) > max(h0, w0)) or min(coord) < 0:
                continue
            cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(save_dir, file_name), image)

    def __call__(self, file_path):
        image = cv2.imread(file_path, 1)
        img = self._image_process(image)
        with torch.no_grad():
            output = self.net(img)[0]
        pred = self._post_process(output)
        if self.conf.test.plot:
            self._plot_rectangle(img, pred, file_path)
        # export torchscript
        self.exportTorchViaScript()
        # if not pred.size(0):
            # return ['None'], torch.Tensor([0])
        scores = pred[:, -2]
        classes = [self.conf.test.idx_to_cls[i] for i in pred[:, -1].long()]
        return classes, scores


if __name__ == "__main__":
    import os
    import sys
    images = sys.argv[1]
    pth = sys.argv[2]

    from config import config
    config.model_path = pth

    det = Yolov5Detection(config)
    for fn in os.listdir(images):
        print("fn: ", fn)
        fp = os.path.join(images, fn)
        res = det(fp)
        print(res)
        break
