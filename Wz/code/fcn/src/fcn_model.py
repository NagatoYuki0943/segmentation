from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101


#---------------------------------------------#
#   重构ResNet50,删除不需要的层,返回Layer3和Layer4的输出
#---------------------------------------------#
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        #---------------------------------------------#
        #   返回的key值是否存在
        #---------------------------------------------#
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        #---------------------------------------------#
        #   将需要的name, new_name强制转换为str     {'layer4': 'out', 'layer3': 'aux'}
        #---------------------------------------------#
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        #---------------------------------------------#
        #   重新构建backbone，将没有使用到的模块全部删掉
        #---------------------------------------------#
        layers = OrderedDict()
        for name, module in model.named_children():
            #---------------------------------------------#
            #   放入全部数据
            #---------------------------------------------#
            layers[name] = module
            #---------------------------------------------#
            #   如果name在里面,就将它从查找dict中删掉
            #---------------------------------------------#
            if name in return_layers:
                del return_layers[name]
            #---------------------------------------------#
            #   全部遍历完就退出
            #---------------------------------------------#
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        #---------------------------------------------#
        #   返回值放入有序字典中
        #---------------------------------------------#
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            #---------------------------------------------#
            #   在要获取的layers中,九田家
            #---------------------------------------------#
            if name in self.return_layers:
                #---------------------------------------------#
                #   找到new_name
                #---------------------------------------------#
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


#---------------------------------------------#
#   FCN主体
#---------------------------------------------#
class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        """
        backbone:       特征提取
        classifier:     主分类器
        aux_classifier  辅助分类器
        """
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        #---------------------------------------------#
        #   取出原图高度和宽度
        #---------------------------------------------#
        input_shape = x.shape[-2:]
        #---------------------------------------------#
        #   特征提取
        #   {'layer4': 'out', 'layer3': 'aux'}
        #---------------------------------------------#
        features = self.backbone(x)

        result = OrderedDict()

        #---------------------------------------------#
        #   主分支
        #---------------------------------------------#
        x = features["out"]
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        #---------------------------------------------#
        #   辅助分支
        #---------------------------------------------#
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        #---------------------------------------------#
        #   {'out': Tensor, 'aux': Tensor}
        #---------------------------------------------#
        return result

#---------------------------------------------#
#   构建分类器
#   3x3Conv -> BN -> ReLU -> Dropout -> 1x1Conv
#   3x3Conv将通道降低为1/4
#   1x1Conv将通道调整为num_classes+1
#---------------------------------------------#
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        #---------------------------------------------#
        #   3x3Conv将通道降低为1/4
        #---------------------------------------------#
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    #---------------------------------------------#
    #   'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    #   'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    #   [False, True, True] Layer3和4使用膨胀卷积
    #---------------------------------------------#
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    #---------------------------------------------#
    #   分支返回的通道和名称
    #---------------------------------------------#
    out_inplanes = 2048
    aux_inplanes = 1024
    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    #---------------------------------------------#
    #   重构ResNet50,删除不需要的层,返回Layer3和Layer4的输出
    #---------------------------------------------#
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    #---------------------------------------------#
    #   辅助分类器默认为None
    #---------------------------------------------#
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)         # head只需要in_channels和out_channels
    #---------------------------------------------#
    #   主分类器
    #---------------------------------------------#
    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model
