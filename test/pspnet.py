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
        pretrained: bool = False,
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
#   PSPModule
#   获取到的特征层划分成不同大小的区域，每个区域内部各自进行平均池化。
#   实现聚合不同区域的上下文信息，从而提高获取全局信息的能力
#   最终将in_channels和四个特征区域的输出合并通道
#                                    in
#                                     │
#       ┌──────────────┬──────────────┼──────────────┬──────────────┐
#       │              │              │              │              │
#       │          AvgPool2d      AvgPool2d      AvgPool2d      AvgPool2d
#       │             1x1            2x2            3x3            6x6
#       │              │              │              │              │
#       │          UpSample       UpSample       UpSample       UpSample
#       │              │              │              │              │
#       │              └─────────────┐│┌─────────────┘              │
#       └─────────────────────────── cat ───────────────────────────┘
#                                     │
#                                  3x3Conv
#                                     │
#                                    out
#-----------------------------------------------------------------------------#
class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pool_sizes: list[int] = [1, 2, 3, 6],
    ) -> None:
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   循环进行设置不同的pool_size
        #-----------------------------------------------------#
        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    ConvNormAct(in_channels, out_channels, 1),
                )
                for pool_size in pool_sizes
            ]
        )

        self.conv_cat = nn.Sequential(
            ConvNormAct(in_channels + out_channels * len(pool_sizes), out_channels, 3),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        pool_outputs = [x]
        for pool in self.pools:
            pool_outputs.append(F.interpolate(pool(x), size=(H, W), mode='bilinear', align_corners=True))
        x = torch.cat(pool_outputs, dim=1)
        x = self.conv_cat(x)
        return x


class PSPNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        downsample_factor: int = 8,
        pool_sizes: list[int] = [1, 2, 3, 6],
        num_classes: int = 21,
        aux: bool = True,
    ) -> None:
        super().__init__()
        self.aux = aux
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
        psp_in_channels = {
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

        #--------------------------------------------------------------#
        #	PSP模块，分区域进行池化,对卷积结果进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   [30, 30, 320] -> [30, 30, 80] -> [30, 30, num_classes]
        #--------------------------------------------------------------#
        psp_out_channels = psp_in_channels // len(pool_sizes)
        self.psp = nn.Sequential(
            PSPModule(psp_in_channels, pool_sizes),
            nn.Conv2d(psp_out_channels, num_classes, 1)         # 分类层
        )

        #----------------------------------#
        #   浅层特征边维度
        #----------------------------------#
        if aux:
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
            aux_out_channels = psp_in_channels // len(pool_sizes) // 2
            self.aux_branch_conv = nn.Sequential(
                ConvNormAct(aux_in_channels, aux_out_channels, 3),
                nn.ReLU(),
                nn.Conv2d(aux_out_channels, num_classes, 1),    # 分类层
            )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # 获得两个特征层
        x, aux_branch = self.backbone(x)

        # psp分支
        x = self.psp(x)
        # 还原为原图大小
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # aux分支
        if self.aux:
            aux_branch = self.aux_branch_conv(aux_branch)
            aux_branch = F.interpolate(aux_branch, size=(H, W), mode='bilinear', align_corners=True)
            return x, aux_branch
        else:
            return x


if __name__ == "__main__":
    x = torch.ones(1, 3, 480, 480)

    model = PSPNet(
        backbone = "resnet18",
        pretrained = True,
        downsample_factor = 8,
        num_classes = 21,
        aux = True,
    )
    model.eval()

    with torch.inference_mode():
        y = model(x)
    if isinstance(y, Tensor):
        print(y.size())
    else:
        for y_ in y:
            print(y_.size())

    if False:
        onnx_path = "pspnet.onnx"
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=["images"],
            output_names=["segmentation"],
            # opset_version=17,
        )
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_model, onnx_path)

