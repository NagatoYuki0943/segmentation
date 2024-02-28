from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


#---------------------------------#
#   Conv+BN+ReLU
#---------------------------------#
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False) # 使用了BN后要设置为False,官方为True
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


#---------------------------------#
#   下采样
#   MaxPool+Conv+BN+ReLU
#---------------------------------#
class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


#---------------------------------#
#   上采样+concat
#   双线性采样+concat+Conv+BN+ReLU
#---------------------------------#
class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): 下层
            x2 (torch.Tensor): 上层
        """
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


#---------------------------------#
#   RSU7 6 5 4
#   有上下采样
#---------------------------------#
class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        """
        Args:
            height (int): REu层数 7 6 5 4
            in_ch (int): in_channel
            mid_ch (int): mid_channel
            out_ch (int): out_channel
        """
        super().__init__()

        assert height >= 2
        #-----------------------------#
        #   第一个conv,输出为out_ch
        #-----------------------------#
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        #--------------------------------------------#
        #   编码器和解码器列表
        #   第一个encoder模块没有下采样,所以flag=False
        #   第一个decoder模块也没有上采样,所以flag=False
        #--------------------------------------------#
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]

        #--------------------------------------------#
        #   具备上下采样层层数为 height - 2
        #   假设是RSU7,en有7个,de有6个,上面en和de都添加了第1个,en最后还有一个,所以这里是7-2个
        #   下采样5次
        #--------------------------------------------#
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))  # 最后输出模块为out_ch,其他为mid_ch

        #--------------------------------------------#
        #   encoder还有最后的一个膨胀卷积
        #--------------------------------------------#
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #-----------------------------#
        #   第一个conv的输出
        #-----------------------------#
        x_in = self.conv_in(x)

        #-----------------------------#
        #   encoder部分
        #-----------------------------#
        x = x_in
        # 保存encoder的全部输出 RSU7一共保存7个值
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        #-----------------------------#
        #   最后encoder的输出,就是膨胀卷积的输出
        #-----------------------------#
        x = encode_outputs.pop()
        # RSU7循环6次,里面也pop6次,上一行也pop一次
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        #------------------#
        #   最后加上x_in
        #------------------#
        return x + x_in


#---------------------------------#
#   RSU7 6 5 4
#   无上下采样
#---------------------------------#
class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        #-----------------------------#
        #   第一个conv,输出为out_ch
        #-----------------------------#
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        #---------------------------------#
        #   encoder,没有下采样,使用膨胀卷积
        #   4层包括中间的层
        #---------------------------------#
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        #---------------------------------#
        #   decoder,没有上采样,使用膨胀卷积
        #---------------------------------#
        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #-----------------------------#
        #   第一个conv的输出
        #-----------------------------#
        x_in = self.conv_in(x)

        #-----------------------------#
        #   encoder部分
        #-----------------------------#
        x = x_in
        # 保存encoder的全部输出 保存4个值
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        #-----------------------------#
        #   最后encoder的输出,就是膨胀卷积的输出
        #-----------------------------#
        x = encode_outputs.pop()
        # 循环3次
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        #------------------#
        #   最后加上x_in
        #------------------#
        return x + x_in


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        """
        Args:
            cfg (dict): 配置
            out_ch (int, optional): 最终通道数,接近1代表为前景概率大,接近0代表背景概率大 Defaults to 1.
        """
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])

        #---------------------------------#
        #   输出对应的3x3Conv
        #---------------------------------#
        side_list = []

        #---------------------------------#
        #   编码器
        #   由上到下
        #---------------------------------#
        encode_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            #---------------------------------#
            #   根据c[4]判断使用RSU还是RSU4F模块
            #---------------------------------#
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            #---------------------------------#
            #   如果收集输出就要添加对应的Conv
            #   编码器最后的e6收集
            #---------------------------------#
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        #---------------------------------#
        #   解码器
        #   由下到上
        #---------------------------------#
        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            #---------------------------------#
            #   根据c[4]判断使用RSU还是RSU4F模块
            #---------------------------------#
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            #---------------------------------#
            #   如果收集输出就要添加对应的Conv
            #   解码器全部收集
            #---------------------------------#
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)

        #---------------------------------#
        #   拼接最终输出后的1x1Conv
        #   in_channel是encoder的长度
        #---------------------------------#
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        #---------------------------------#
        #   收集encoder输出
        #---------------------------------#
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            # 如果不是最后的encoder要进行下采样
            if i != self.encode_num - 1:    # 6 - 1 = 5
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        #---------------------------------#
        #   收集decoder输出
        #---------------------------------#
        x = encode_outputs.pop()
        #---------------------------------#
        #   收集最终的输出,e6的输出
        #---------------------------------#
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.concat([x, x2], dim=1))
            #---------------------------------#
            #   插入到最前面,因此最终decode_outputs的数据顺序为 d1 d2 d3 d4 d5 e6 的输出
            #---------------------------------#
            decode_outputs.insert(0, x)

        #-----------------------------------------------------------------------------#
        #   收集侧边的输出
        #   side_modules建立的顺序是     e6 d5 d4 d3 d2 d1
        #   上面decode_outputs倒叙插入为 d1 d2 d3 d4 d5 e6, pop从尾部pop,所以正好对应
        #-----------------------------------------------------------------------------#
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        #-----------------------#
        #   最终卷积通道为1
        #-----------------------#
        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            # [x, sub1, sub2, sub3, sub4, sub5, sub6]
            return [x] + side_outputs
        else:
            return torch.sigmoid(x)


def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side(是否收集decoder的输出)
        "encode": [[7,   3,  32,  64, False, False],   # En1    RSU7
                   [6,  64,  32, 128, False, False],   # En2    RSU6
                   [5, 128,  64, 256, False, False],   # En3    RSU5
                   [4, 256, 128, 512, False, False],   # En4    RSU4
                   [4, 512, 256, 512,  True, False],   # En5    RSU4F
                   [4, 512, 256, 512,  True, True]],   # En6    RSU4F
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512,  True, True],   # De5    RSU4F
                   [4, 1024, 128, 256, False, True],   # De4    RSU4
                   [5,  512,  64, 128, False, True],   # De3    RSU5
                   [6,  256,  32,  64, False, True],   # De2    RSU6
                   [7,  128,  16,  64, False, True]]   # De1    RSU7
    }

    return U2Net(cfg, out_ch)


def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16,  64, False, False],  # En1    RSU7
                   [6, 64, 16, 64, False, False],  # En2    RSU6
                   [5, 64, 16, 64, False, False],  # En3    RSU5
                   [4, 64, 16, 64, False, False],  # En4    RSU4
                   [4, 64, 16, 64, True,  False],  # En5    RSU4F
                   [4, 64, 16, 64, True,  True]],  # En6    RSU4F
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True,  True],  # De5    RSU4F
                   [4, 128, 16, 64, False, True],  # De4    RSU4
                   [5, 128, 16, 64, False, True],  # De3    RSU5
                   [6, 128, 16, 64, False, True],  # De2    RSU6
                   [7, 128, 16, 64, False, True]]  # De1    RSU7
    }

    return U2Net(cfg, out_ch)


def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    # u2net = u2net_full()
    # convert_onnx(u2net, "u2net_full.onnx")

    x = torch.randn(1, 3, 512, 512)
    model = u2net_lite()
    model.eval()
    y = model(x)
    print(y.size()) # [1, 1, 512, 512]