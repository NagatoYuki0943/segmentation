"""refer https://github.com/bubbliiiing/deeplabv3-plus-pytorch/blob/main/nets/deeplabv3_plus.py
"""

from functools import partial
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchsummary import summary


def _nostride_dilate(m, dilate):
    """
    https://github.com/bubbliiiing/deeplabv3-plus-pytorch/blob/main/nets/deeplabv3_plus.py#L33

    m: apply中的一个模型实例
    dilate: apply中的自定义参数，扩张系数
    """
    if isinstance(m, nn.Conv2d):
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilate//2, dilate//2)
                m.padding = (dilate//2, dilate//2)
        elif m.kernel_size == (3, 3):
            m.dilation = (dilate, dilate)
            m.padding = (dilate, dilate)


class ResNet(nn.Module):
    def __init__(
        self,
        version: str = "resnet18",
        pretrained: bool = True,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        model = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "resnext50_32x4d": models.resnext50_32x4d,
            "resnext101_64x4d": models.resnext101_64x4d,
            "resnext101_32x8d": models.resnext101_32x8d,
        }[version]
        weight = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
            "resnet152": models.ResNet152_Weights.DEFAULT,
            "resnext50_32x4d": models.ResNeXt50_32X4D_Weights.DEFAULT,
            "resnext101_64x4d": models.ResNeXt101_64X4D_Weights.DEFAULT,
            "resnext101_32x8d": models.ResNeXt101_32X8D_Weights.DEFAULT,
        }[version]
        model = model(weights=weight if pretrained else None)
        del model.avgpool
        del model.fc

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # 调整下采样倍率
        if downsample_factor == 8:
            self.layer3.apply(partial(_nostride_dilate, dilate=2))
            self.layer4.apply(partial(_nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            self.layer4.apply(partial(_nostride_dilate, dilate=2))

    def forward(self, x: Tensor) -> tuple[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        aux_branch = x      # 第2次下采样为aux_branch 1/4 [H, W]
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, aux_branch


class MobileNetV2(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.features = model.features[:-1]
        self.total_idx  = len(self.features)    # 18
        self.down_idx   = [0, 2, 4, 7, 14]      # 下采样层index

        # 调整下采样倍率
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(_nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(_nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(_nostride_dilate, dilate=2))

    def forward(self, x: Tensor) -> tuple[Tensor]:
        x = self.features[0:self.down_idx[2]](x)   # 第2次下采样为aux_branch 1/4 [H, W]
        aux_branch = x
        x = self.features[self.down_idx[2]:](x)
        return x, aux_branch


class MobileNetV3(nn.Module):
    def __init__(
        self,
        version: str = "mobilenet_v3_large",
        pretrained: bool = True,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        model = {
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }[version]
        weight = {
            "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
            "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        }[version]
        model = model(weights=weight if pretrained else None)
        self.features = model.features[:-1]
        self.total_idx  = len(self.features)    # 12 16
        self.down_idx   = {                     # 下采样层index
            "mobilenet_v3_small": [0, 1, 2, 4, 9],
            "mobilenet_v3_large": [0, 2, 4, 7, 13],
        }[version]

        # 调整下采样倍率
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(_nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(_nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(_nostride_dilate, dilate=2))

    def forward(self, x: Tensor) -> tuple[Tensor]:
        x = self.features[0:self.down_idx[2]](x)   # 第2次下采样为aux_branch 1/4 [H, W]
        aux_branch = x
        x = self.features[self.down_idx[2]:](x)
        return x, aux_branch


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        version: str = "shufflenet_v2_x1_0",
        pretrained: bool = True,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        model = {
            "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
            "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
            "shufflenet_v2_x1_5": models.shufflenet_v2_x1_5,
            "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0,
        }[version]
        weight = {
            "shufflenet_v2_x0_5": models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
            "shufflenet_v2_x1_0": models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
            "shufflenet_v2_x1_5": models.ShuffleNet_V2_X1_5_Weights.DEFAULT,
            "shufflenet_v2_x2_0": models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
        }[version]
        model = model(weights=weight if pretrained else None)

        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4

        # 调整下采样倍率
        if downsample_factor == 8:
            self.stage3.apply(partial(_nostride_dilate, dilate=2))
            self.stage4.apply(partial(_nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            self.stage4.apply(partial(_nostride_dilate, dilate=2))

    def forward(self, x: Tensor) -> tuple[Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)
        aux_branch = x      # 第2次下采样为aux_branch 1/4 [H, W]
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x, aux_branch


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = norm(out_channels)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norm(self.conv(x)))
        return x


#-----------------------------------------------------------------------------#
#   ASPP
#   压缩四次的初步有效特征层利用并行的Atrous Convolution，
#   分别用不同rate的Atrous Convolution进行特征提取，再进行合并，再进行1x1卷积压缩特征。
#                                    in
#                                     │
#       ┌──────────────┬──────────────┼──────────────┬──────────────┐
#       │              │              │              │              │
#    1x1Conv        3x3Conv        3x3Conv        3x3Conv   AdaptiveAvgPool2d
#       │          dilation=6    dilation=12    dilation=18        1x1
#       │              │              │              │              │
#       │              │              │              │           1x1Conv
#       │              │              │              │              │
#       │              │              │              │           UpSample
#       │              └─────────────┐│┌─────────────┘              │
#       └─────────────────────────── cat ───────────────────────────┘
#                                     │
#                                  1x1Conv
#                                     │
#                                    out
#-----------------------------------------------------------------------------#
class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rate: int = 1,
    ) -> None:
        super().__init__()
        self.branch1 = ConvNormAct(in_channels, out_channels, 1)
        self.branch2 = ConvNormAct(in_channels, out_channels, 3, padding=6*rate, dilation=6*rate)
        self.branch3 = ConvNormAct(in_channels, out_channels, 3, padding=12*rate, dilation=12*rate)
        self.branch4 = ConvNormAct(in_channels, out_channels, 3, padding=18*rate, dilation=18*rate)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(in_channels, out_channels, 1),
        )

        # 拼接后的1x1卷积,聚合信息
        self.conv_cat = ConvNormAct(5*out_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()

        conv1x1   = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, size=(H, W), mode='bilinear', align_corners=True)

        x = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        x = self.conv_cat(x)
        return x


class Deeplab(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        downsample_factor: int = 8,
        num_classes: int = 21,
    ) -> None:
        super().__init__()
        if "resnet" in backbone:
            self.backbone = ResNet(backbone, pretrained, downsample_factor)
        elif "mobilenet_v2" == backbone:
            self.backbone = MobileNetV2(pretrained, downsample_factor)
        elif "mobilenet_v3" in backbone:
            self.backbone = MobileNetV3(backbone, pretrained, downsample_factor)
        elif "shufflenet_v2" in backbone:
            self.backbone = ShuffleNetV2(backbone, pretrained, downsample_factor)
        else:
            raise ValueError(f"unsupported backbone {backbone}")

        #------------------#
        #   分支输出通道处
        #------------------#
        aspp_in_channels = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048,
            "resnext50_32x4d": 2048,
            "resnext101_64x4d": 2048,
            "resnext101_32x8d": 2048,
            "mobilenet_v2": 320,
            "mobilenet_v3_small": 96,
            "mobilenet_v3_large": 160,
            "shufflenet_v2_x0_5": 192,
            "shufflenet_v2_x1_0": 464,
            "shufflenet_v2_x1_5": 704,
            "shufflenet_v2_x2_0": 976,
        }[backbone]
        aux_in_channels = {
            "resnet18": 64,
            "resnet34": 64,
            "resnet50": 256,
            "resnet101": 256,
            "resnet152": 256,
            "resnext50_32x4d": 256,
            "resnext101_64x4d": 256,
            "resnext101_32x8d": 256,
            "mobilenet_v2": 24,
            "mobilenet_v3_small": 16,
            "mobilenet_v3_large": 24,
            "shufflenet_v2_x0_5": 24,
            "shufflenet_v2_x1_0": 24,
            "shufflenet_v2_x1_5": 24,
            "shufflenet_v2_x2_0": 24,
        }[backbone]

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取,获取深层特征
        #-----------------------------------------#
        aspp_out_channels = aux_in_channels * 2 # 自己设定的
        self.aspp = ASPP(aspp_in_channels, aspp_out_channels)

        #----------------------------------#
        #   浅层特征边维度变化 1层1x1Conv
        #----------------------------------#
        aux_out_channels = aux_in_channels      # 自己设定的
        self.aux_branch_conv = ConvNormAct(aux_in_channels, aux_out_channels, 1)

        #-----------------------------------------#
        #   拼接后的特征提取 2层3x3Conv
        #-----------------------------------------#
        cat_out_channels = aux_in_channels * 2  # 自己设定的
        self.cat_conv = nn.Sequential(
            ConvNormAct(aspp_out_channels + aux_out_channels, cat_out_channels, 3),
            nn.Dropout(0.5),
            ConvNormAct(cat_out_channels, cat_out_channels, 3),
        )

        self.cls_conv = nn.Conv2d(cat_out_channels, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        #-----------------------------------------#
        #   获得两个特征层
        #   aux_branch: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        x, aux_branch = self.backbone(x)
        aux_branch = self.aux_branch_conv(aux_branch)
        x = self.aspp(x)

        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        _, _, auxH, auxW = aux_branch.shape
        x = F.interpolate(x, size=(auxH, auxW), mode='bilinear', align_corners=True)
        x = torch.cat([x, aux_branch], dim=1)
        x = self.cat_conv(x)
        x = self.cls_conv(x)

        # 还原为原图大小
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    x = torch.ones(1, 3, 512, 512)
    model = Deeplab(
        backbone = "mobilenet_v3_large",
        pretrained = True,
        downsample_factor = 8,
        num_classes = 21,
    )
    model.eval()

    with torch.inference_mode():
        y = model(x)
    print(y.size())

    if True:
        onnx_path = "deeplabv3_plus.onnx"
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=["images"],
            output_names=["segmentation"],
            opset_version=17,
        )
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_model, onnx_path)

