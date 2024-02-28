"""使用ResNet残差块构建Unet
"""

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchsummary import summary


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


class BasicBlock(nn.Module):
    """不标准的ResNet BasicBlock"""
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()

        self.conv1 = ConvNormAct(in_channels, channels, 3, stride=stride, groups=groups, norm=norm, act=act)
        self.conv2 = ConvNormAct(channels, channels, 3, groups=groups, norm=norm, act=act)

        if in_channels != channels or stride != 1:
            self.shortcut = ConvNormAct(in_channels, channels, 3, stride=stride, groups=groups, norm=norm, act=act)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        x = self.conv2(self.conv1(x))
        x = x + shortcut
        return x


class Bottleneck(nn.Module):
    """不标准的ResNet Bottleneck"""
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_channel = int(channels * (base_width / 64.0)) * groups
        out_channels = channels * self.expansion

        self.conv1 = ConvNormAct(in_channels, hidden_channel, 1, groups=groups, norm=norm, act=act)
        self.conv2 = ConvNormAct(hidden_channel, hidden_channel, 3, stride=stride, groups=groups, norm=norm, act=act)
        self.conv3 = ConvNormAct(hidden_channel, out_channels, 1, groups=groups, norm=norm, act=act)

        if in_channels != out_channels or stride != 1:
            self.shortcut = ConvNormAct(in_channels, out_channels, 3, stride=stride, groups=groups, norm=norm, act=act)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x + shortcut
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dims: list[int] = [32, 64, 128, 256, 512],
        block_repeats: int = 3, # 每个stage下采样次数,包含下采样层
        block: nn.Module = BasicBlock,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        in_channels = dims[0]

        self.encoders = nn.ModuleList()
        for i, dim in enumerate(dims):
            encoder = []
            # 下采样层,第一次不下采样
            if i != 0:
                encoder.append(block(in_channels, dim, stride=2, norm=norm, act=act))
                in_channels = dim * block.expansion
            for _ in range(block_repeats - 1):
                encoder.append(
                    block(in_channels, dim, norm=norm, act=act)
                )
                in_channels = dim * block.expansion
            self.encoders.append(nn.Sequential(*encoder))

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            outputs.append(x)
        return outputs
        # [1, 32, 512, 512]
        # [1, 64, 256, 256]
        # [1, 128, 128, 128]
        # [1, 256, 64, 64]
        # [1, 512, 32, 32]


#------------------------------------------------------------------------------------------------------#
#   上采样 + 拼接
#   [1, 512, 32, 32] -> [1, 512, 64, 64] -> [1, 256, 64, 64] cat [1, 256, 64, 64] = [1, 512, 64, 64]
#------------------------------------------------------------------------------------------------------#
class UpSampleCat(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = ConvNormAct(in_channels, out_channels, 1)

    def forward(self, low: Tensor, high: Tensor) -> Tensor:
        low = self.upsample(low)
        low = self.conv(low)
        x = torch.cat((low, high), dim=1)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dims: list[int] = [32, 64, 128, 256, 512],
        block_repeats: int = 3, # 每个stage下采样次数,包含通道变换层
        block: nn.Module = BasicBlock,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        dims = dims[::-1]

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # [1024, 512, 256, 128, 64]
        for i in range(len(dims) - 1):
            # 0 1 2 3
            in_channels = dims[i] * block.expansion
            channels = dims[i+1] * block.expansion
            self.upsamples.append(UpSampleCat(in_channels, channels))
            decoder = []
            decoder.append(block(in_channels, dims[i+1], norm=norm, act=act))  # 通道变化层 512 -> 256
            for _ in range(block_repeats - 1):
                decoder.append(
                    block(channels, dims[i+1], norm=norm, act=act)
                )
            self.decoders.append(nn.Sequential(*decoder))

    def forward(self, x: list[Tensor]) -> Tensor:
        low = x[-1]
        # 倒序的切片开始和结束也要从右到左数
        for i, high in enumerate(x[-2::-1]):    # [-2::-1] 倒序忽略最后一个
            low = self.upsamples[i](low, high)
            low = self.decoders[i](low)
        return low


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dims: list[int] = [32, 64, 128, 256, 512],
        block_repeats: int = 3, # 每个stage下采样次数,包含下采样层
        num_classes: int = 21,
        block: nn.Module = BasicBlock,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.in_conv = ConvNormAct(in_channels, dims[0], 3)
        self.encoder = Encoder(dims, block_repeats, block=block, norm=norm, act=act)
        self.decoder = Decoder(dims, block_repeats, block=block, norm=norm, act=act)
        self.out_conv = nn.Conv2d(dims[0] * block.expansion, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    x = torch.ones(1, 3, 512, 512)
    model = Unet(
        block_repeats=3,
        num_classes=21,
        block=BasicBlock,
    )
    model.eval()

    with torch.inference_mode():
        y = model(x)
    print(y.size())

    if False:
        onnx_path = "unet.onnx"
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
