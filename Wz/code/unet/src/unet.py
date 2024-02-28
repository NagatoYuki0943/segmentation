from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

#---------------------------------------------#
#  两个3x3Conv+BN+ReLU
#---------------------------------------------#
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


#---------------------------------------------#
#   2倍下采样
#   MaxPooling + 2个3x3Conv
#---------------------------------------------#
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

#---------------------------------------------#
#   2倍上采样+拼接+2个3x3Conv
#---------------------------------------------#
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        in_channels:  上采样拼接后的channels
        out_channels: 输出channels
        """
        super(Up, self).__init__()
        #---------------------------------------------#
        #   使用双线性插值代替转置卷积
        #---------------------------------------------#
        if bilinear:
            #---------------------------------------------#
            #   双线性插值上采样后通道不变,两个卷积中第一个卷积让通道减半,第二个卷积又减半
            #   为了让最后的通道能和更浅层的特征通道数一致
            #---------------------------------------------#
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) # in_channels // 2 降低middle_channel
        else:
            #---------------------------------------------#
            #   转置卷积后直接将通道减半
            #---------------------------------------------#
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)    # 直接将通道变为 in_channels // 2
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: 深层特征
        x2: 浅色特征
        """
        #---------------------------------------------#
        #   上采样
        #---------------------------------------------#
        x1 = self.up(x1)


        #---------------------------------------------#
        #   对上采样的x1进行padding,防止出现图片大小不是16倍数
        #   [N, C, H, W]
        #   上层图片宽高 - 上采样后的宽高 = 差值
        #---------------------------------------------#
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        #---------------------------------------------#
        #   padding_left, padding_right, padding_top, padding_bottom
        #---------------------------------------------#
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        #---------------------------------------------#
        #   两层通道拼接后
        #---------------------------------------------#
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

#---------------------------------------------#
#   最后1x1Conv将通道数调整为num_classes+1
#---------------------------------------------#
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


#---------------------------------------------#
#   UNet搭建
#---------------------------------------------#
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,  # 默认单通道
                 num_classes: int = 2,  # num_classes+1
                 bilinear: bool = True, # 使用双线性插值进行上采样
                 base_c: int = 64):     # 第一个卷积的out_channel,后面的会翻倍
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        #---------------------------------------------#
        # 开始的2个卷积
        #---------------------------------------------#
        self.in_conv = DoubleConv(in_channels, base_c)
        #---------------------------------------------#
        #   下采样部分
        #---------------------------------------------#
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        #---------------------------------------------#
        #   双线性插值的最后下采样通道数没有翻倍,上一层输出为为512,这一层上采样后也为512,正好相同
        #---------------------------------------------#
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        #---------------------------------------------#
        #   上采样部分,通道减半
        #---------------------------------------------#
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        #---------------------------------------------#
        #   最后1x1Conv将通道数调整为num_classes+1
        #---------------------------------------------#
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #---------------------------------------------#
        #   上采样部分 参数为深层,浅层
        #---------------------------------------------#
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        #---------------------------------------------#
        #   return {"out": logits}
        #---------------------------------------------#
        return {"out": logits}
