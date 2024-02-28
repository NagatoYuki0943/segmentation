from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .mobilenet_backbone import mobilenet_v3_large


#---------------------------------------------#
#   重构模型
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


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """
    __constants__ = ['aux_classifier']

    def __init__(self,
                 backbone: nn.Module,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int = 128) -> None:
        super(LRASPP, self).__init__()
        #---------------------------------------------#
        #   return {"low":Tensor, "high":Tensor}
        #---------------------------------------------#
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        #---------------------------------------------#
        #   return {"low":Tensor, "high":Tensor}
        #---------------------------------------------#
        features = self.backbone(x)
        #---------------------------------------------#
        #   return Tensor
        #---------------------------------------------#
        out = self.classifier(features)
        #---------------------------------------------#
        #   还原到原图尺寸
        #---------------------------------------------#
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


#---------------------------------------------#
#   分割头部分
#   分支1:
#   ​	1x1Conv+BN+ReLU 通道变为128
#   分支2(通道注意力):
#   ​	全局平均池化 -> 1x1Conv 通道变为128 -> sigmoid -> 和上面相乘即可 (上采样部分不需要)
#   分支相乘 -> 上采样 -> 1x1Conv调整通道数为num_classes
#   下采样8倍的数据引过来 -> 1x1Conv调整通道数为num_classes
#   分支合并的结果和下采样8倍的结果相加最后上采样8倍得到最后结果
#---------------------------------------------#
class LRASPPHead(nn.Module):
    def __init__(self,
                 low_channels: int,             # 8倍下采样的out_channels
                 high_channels: int,            # 16倍下采样的out_channels
                 num_classes: int,
                 inter_channels: int) -> None:  # 降低通道数
        super(LRASPPHead, self).__init__()
        #---------------------------------------------#
        #   分支1:
        #   ​	1x1Conv+BN+ReLU 通道变为128
        #---------------------------------------------#
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        #---------------------------------------------#
        #   分支2(通道注意力):
        #   ​	全局平均池化 -> 1x1Conv 通道变为128 -> sigmoid -> 和上面相乘即可 (上采样部分不需要)
        #---------------------------------------------#
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )
        #---------------------------------------------#
        #   下采样8倍的数据引过来 -> 1x1Conv调整通道数为num_classes
        #---------------------------------------------#
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        #---------------------------------------------#
        #   分支合并 -> 上采样 -> 1x1Conv调整通道数为num_classes
        #---------------------------------------------#
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        #---------------------------------------------#
        #   {"low":Tensor, "high":Tensor}
        #   low: 8倍下采样, high: 16倍下采样
        #---------------------------------------------#
        low = inputs["low"]
        high = inputs["high"]
        #---------------------------------------------#
        #   分支1:
        #   ​	1x1Conv+BN+ReLU 通道变为128
        #   分支2(通道注意力):
        #   ​	全局平均池化 -> 1x1Conv 通道变为128 -> sigmoid -> 和上面相乘即可 (上采样部分不需要)
        #   分支1 * 分支2
        #---------------------------------------------#
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        #---------------------------------------------#
        #   深层特征上采样2倍到低层宽高
        #---------------------------------------------#
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        #---------------------------------------------#
        #   深浅特征相加
        #---------------------------------------------#
        return self.low_classifier(low) + self.high_classifier(x)


def lraspp_mobilenetv3_large(num_classes=21, pretrain_backbone=False):
    #---------------------------------------------#
    #   'mobilenetv3_large_imagenet': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    #   'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth'
    #---------------------------------------------#
    backbone = mobilenet_v3_large(dilated=True) # 最后三个卷积使用了扩张卷积

    if pretrain_backbone:
        # 载入mobilenetv3 large backbone预训练权重
        backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

    #---------------------------------------------#
    #   取出特征提取部分
    #---------------------------------------------#
    backbone = backbone.features

    #---------------------------------------------#
    #   Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    #   The first and last blocks are always included because they are the C0 (conv1) and Cn.
    #   遍历features取到有 "is_strided" 的结构id(步长为2的MBlock索引)
    #   [len(backbone) - 1] 最后1x1Conv的索引
    #---------------------------------------------#
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]     # use C2 here which has output_stride = 8   下采样8倍索引
    high_pos = stage_indices[-1]    # use C5 which has output_stride = 16       下采样16倍索引
    low_channels = backbone[low_pos].out_channels                               #下采样8倍的out_channels
    high_channels = backbone[high_pos].out_channels                             #下采样16倍的out_channels

    #---------------------------------------------#
    #   返回2层,下采样8倍和16倍的两层
    #   {"low":Tensor, "high":Tensor}
    #---------------------------------------------#
    return_layers = {str(low_pos): "low", str(high_pos): "high"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model
