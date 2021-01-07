# Modified from ultralytics/yolov5 by Zhiqiang Wang
import json
import torch
from torch import nn, Tensor
from collections import OrderedDict
from typing import List, Dict, Optional

from .model_base import *
from .utils_anchor import AnchorGenerator


__all__ = ['YoloV5']

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers: Dict[str, str], save_list: List[Tensor]):
        self.return_layers = return_layers.copy()
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.save_list = save_list

    def forward(self, x: Tensor):
        out = OrderedDict()
        y: List[Tensor] = []

        for i, (name, module) in enumerate(self.items()):
            if module.f > 0:  # Concat layer
                x = torch.cat([x, y[sorted(self.save_list).index(module.f)]], 1)
            else:
                x = module(x)  # run
            if i in self.save_list:
                y.append(x)  # save output

            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class YoloBody(nn.Module):
    def __init__(self, layers, save_list: List[int]):
        super().__init__()
        self.model = nn.Sequential(*layers)
        self.save_list = save_list
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        y: List[Tensor] = []

        for i, m in enumerate(self.model):
            if m.f > 0:  # Concat layer
                out = torch.cat([out, y[sorted(self.save_list).index(m.f)]], 1)
            else:
                out = m(out)  # run
            if i in self.save_list:
                y.append(out)  # save output
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True


class YoloBackbone(nn.Module):
    def __init__(self, yolo_body: nn.Module, return_layers: dict, out_channels: List[int]):
        super().__init__()
        self.body = IntermediateLayerGetter(yolo_body.model, return_layers=return_layers, save_list=yolo_body.save_list)
        self.out_channels = out_channels

    def forward(self, x: Tensor):
        out: List[Tensor] = []

        x = self.body(x)
        for name, feature in x.items():
            out.append(feature)
        return out


# YoloV5 head
class Detect(nn.Module):
    def __init__(self, in_channels: List[int], anchor_grids: List[int], num_classes: int):
        super().__init__()
        self.nl = len(anchor_grids)  # anchors
        self.na = len(anchor_grids[0]) // 2
        self.anchors = torch.tensor(anchor_grids).float().view(self.nl, -1, 2)
        self.no = num_classes + 5  # number of outputs per anchor
        self.head = nn.ModuleList(nn.Conv2d(ch, self.no * self.nl, 1) for ch in in_channels)  # output conv

    def get_result_from_head(self, features: Tensor, idx: int) -> Tensor:
        num_blocks = 0
        for m in self.head:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = features
        for module in self.head:
            if i == idx:
                out = module(features)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        all_pred_logits: List[Tensor] = []  # inference output

        for i, features in enumerate(x):
            pred_logits = self.get_result_from_head(features, i)
            # Permute output from (N, A * K, H, W) to (N, A, H, W, K)
            N, _, H, W = pred_logits.shape
            pred_logits = pred_logits.view(N, self.nl, -1, H, W)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2)  # Size=(N, A, H, W, K)
            all_pred_logits.append(pred_logits)
        return all_pred_logits


class YoloV5(nn.Module):
    is_training = True
    def __init__(self, model_file: str, model_type: str, num_classes: int, strides: List[int]):
        super().__init__()
        # parse model file
        with open(model_file, 'r') as f:
            model_dict = json.load(f)
            model_dict["depth_multiple"] = model_dict["depth_multiple"][model_type.lower()]
            model_dict["width_multiple"] = model_dict["width_multiple"][model_type.lower()]
        layers, save_list, head_info = parse_model(model_dict, in_channels=3)
        out_channels=head_info[0]
        anchor_grids = head_info[1]
        return_layers= {str(key): str(i) for i, key in enumerate(head_info[2])}
        # backbone: darknet
        body = YoloBody(layers, save_list)
        self.backbone = YoloBackbone(body, return_layers, out_channels)
        self.anchor_generator = AnchorGenerator(strides, anchor_grids)
        self.nc = num_classes
        self.head = Detect(out_channels, self.anchor_generator.anchor_grids, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        head_outputs = self.head(features)
        if self.is_training:
            return head_outputs
        anchors_tuple = self.anchor_generator(features)
        N, _, _, _, K = head_outputs[0].shape
        all_pred_logits: List[Tensor] = []
        for pred_logits in head_outputs:
            pred_logits = pred_logits.reshape(N, -1, K)  # Size=(N, HWA, K)
            all_pred_logits.append(pred_logits)
        preds = torch.cat(all_pred_logits, dim=1)
        preds = torch.sigmoid(preds)
        preds[..., 0:2] = (preds[..., 0:2] * 2. + anchors_tuple[0]) * anchors_tuple[1]
        preds[..., 2:4] = (preds[..., 2:4] * 2) ** 2 * anchors_tuple[2]
        return preds


def parse_model(model_dict, in_channels=3):
    head_info = ()
    anchors, num_classes = model_dict['anchors'], model_dict['nc']
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (num_classes + 5)

    c2 = in_channels
    layers, save_list, channels = [], [], [c2]  # layers, save list, channels out
    # from, number, module, args
    for i, (f, n, m, args) in enumerate(model_dict['backbone'] + model_dict['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * model_dict['depth_multiple']), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = channels[f], args[0]
            c2 = _make_divisible(c2 * model_dict['width_multiple'], 8) if c2 != num_outputs else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [channels[f]]
        elif m is Concat:
            c2 = sum([channels[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            num_layers, anchor_grids = f, args[-1]
            out_channels = [channels[x + 1] for x in f]
            head_info = (out_channels, anchor_grids, num_layers)
            continue
        else:
            c2 = channels[f]

        module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        module.f = -1 if f == -1 else f[-1]

        save_list.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(module)
        channels.append(c2)
    return layers, save_list, head_info


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


