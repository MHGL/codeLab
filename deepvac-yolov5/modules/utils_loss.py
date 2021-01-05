#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/05
@Desciption : deepvac yolov5 loss, include lbox, lobj, lcls, refer to  https://github.com/ultralytics/yolov5/blob/master/utils/loss.py
'''


import math
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  
            return loss


class MultiLoss(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config.loss
        self.device = deepvac_config.device

    def _smooth_BCE(self, eps=0.1): 
        return 1.0 - 0.5 * eps, 0.5 * eps

    def _bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
        box2 = box2.T
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  
            if CIoU or DIoU:  
                c2 = cw ** 2 + ch ** 2 + eps 
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4 
                if DIoU:
                    return iou - rho2 / c2  
                elif CIoU:  
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / ((1 + eps) - iou + v)
                    return iou - (rho2 / c2 + v * alpha) 
            else:  
                c_area = cw * ch + eps  
                return iou - (c_area - union) / c_area  
        else:
            return iou  
            
    def _build_targets(self, preds, targets, model):
        '''
            Build targets for compute_loss() 
            input preds: [Tensor(1, 3, 13, 13, 9), Tensor(1, 3, 26, 26, 9), Tensor(1, 3, 52, 52, 9)]
            input targets(image,class,x,y,w,h)
        '''
        det = model.head
        na, nt = det.na, targets.shape[0]  
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device) 
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  
        g = 0.5  
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  
                            ], device=targets.device).float() * g  
        for i in range(det.nl):
            anchors = det.anchors[i].to(targets.device)
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]] 
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None] 
                j = torch.max(r, 1. / r).max(2)[0] < self.conf.anchor_t
                t = t[j] 
                gxy = t[:, 2:4]  
                gxi = gain[[2, 3]] - gxy  
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4] 
            gwh = t[:, 4:6] 
            gij = (gxy - offsets).long()
            gi, gj = gij.T  
            a = t[:, 6].long()  
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  
            tbox.append(torch.cat((gxy - gij, gwh), 1)) 
            anch.append(anchors[a])  
            tcls.append(c) 
        return tcls, tbox, indices, anch

    def __call__(self, preds, targets, model):  
        device = self.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self._build_targets(preds, targets, model)  
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.conf.cls_pw], device=device)) 
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.conf.obj_pw], device=device))
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = self._smooth_BCE(eps=0.0)
        # Focal loss
        g = self.conf.fl_gamma  
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        # Losses
        nt = 0  
        no = len(preds)
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  
        iou_list = []
        for i, pi in enumerate(preds): 
            b, a, gj, gi = indices[i] 
            tobj = torch.zeros_like(pi[..., 0], device=device)  
            n = b.shape[0]  
            if n:
                nt += n  
                ps = pi[b, a, gj, gi]  
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  
                iou = self._bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  
                iou_list.append(iou.mean().item())
                lbox += (1.0 - iou).mean()  
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.conf.gr) + self.conf.gr * iou.detach().clamp(0).type(tobj.dtype)  
                # Classification
                if model.nc > 1:
                    t = torch.full_like(ps[:, 5:], cn, device=device)  
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t) 
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  
        s = 3 / no 
        lbox *= self.conf.box * s
        lobj *= self.conf.obj * s * (1.4 if no == 4 else 1.)
        lcls *= self.conf.cls * s
        bs = tobj.shape[0]  
        iou = sum(iou_list) / len(iou_list) if iou_list else 0
        # loss = 0.2 * lbox + 0.5 * lobj + 0.3 * lcls
        return lbox, lobj, lcls, iou


if __name__ == "__main__":
    import torch

    from config import config
    from modules.model import YoloV5

    loss_fn = MultiLoss(config)
    net = YoloV5(config.model_file,
                        config.model_type,
                        config.num_classes,
                        config.strides).to(config.device)
    net.is_training = True
    
    x = torch.randn((3, 3, 416, 416)).to(config.device)
    t = torch.Tensor([
                    [0, 0, .2, .2, .2, .2],
                    [1, 1, .1, .1, .1, .1],
                    [1, 3, .2, .7, .5, .6],
                    [1, 3, .1, .2, .3, .4],
                    ]).to(config.device)
    preds = net(x)
    loss = loss_fn(preds, t, net)
    print("loss", loss)
