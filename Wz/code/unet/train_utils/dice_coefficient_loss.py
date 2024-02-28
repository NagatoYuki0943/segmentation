import torch
import torch.nn as nn


#---------------------------------------------#
#   每个类别都要有GT,对于背景,指定类别都要求
#---------------------------------------------#
def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """
    build target for dice coefficient

    ignore_index: 忽略的颜色, 默认 -100 不使用, 这里忽略了255
    """
    dice_target = target.clone()
    if ignore_index >= 0:
        #---------------------------------------------#
        #   找255的位置
        #---------------------------------------------#
        ignore_mask = torch.eq(target, ignore_index)
        #---------------------------------------------#
        #   255区域变为0
        #---------------------------------------------#
        dice_target[ignore_mask] = 0
        #---------------------------------------------#
        #   转换成one-hot编码 背景: 10 前景: 01
        #   [N, H, W] -> [N, H, W, C]
        #---------------------------------------------#
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        #---------------------------------------------#
        #   255区域再变为255,计算损失忽略掉
        #---------------------------------------------#
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    # [N, H, W, C] -> [N, C, H, W]
    return dice_target.permute(0, 3, 1, 2)


#---------------------------------------------#
#   Average of Dice coefficient for all batches, or for a single mask
#   计算一个batch中所有图片某个类别的dice_coefficient
#---------------------------------------------#
def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """
    x:      每个类别预测
    target: 每个真实target
    ignore_index: 忽略哪个index 使用了255
    """
    d = 0.
    batch_size = x.shape[0]
    #---------------------------------------------#
    #   遍历没一张图片
    #---------------------------------------------#
    for i in range(batch_size):
        #---------------------------------------------#
        #   第i张图片展平
        #---------------------------------------------#
        x_i = x[i].reshape(-1)
        #---------------------------------------------#
        #   第i个target
        #---------------------------------------------#
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            #---------------------------------------------#
            #   找出mask中不为ignore_index的区域
            #---------------------------------------------#
            roi_mask = torch.ne(t_i, ignore_index)
            #---------------------------------------------#
            #   找到预测值中和target中感兴趣的区域
            #---------------------------------------------#
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        #---------------------------------------------#
        #   分子,分母
        #---------------------------------------------#
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        #---------------------------------------------#
        #   分母为0,说明预测值全为0,真实值全为0,说明全预测对了,设为2倍inter,相除之后为1
        #---------------------------------------------#
        if sets_sum == 0:
            sets_sum = 2 * inter

        #---------------------------------------------#
        #   dice损失计算公式
        #---------------------------------------------#
        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


#---------------------------------------------#
#   分别计算每个类别的损失
#   Average of Dice coefficient for all classes
#---------------------------------------------#
def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """
    x:      全部类别预测
    target: 全部真实target
    """
    dice = 0.
    #---------------------------------------------#
    #   遍历每个channel(类别)计算dice
    #---------------------------------------------#
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)
    #---------------------------------------------#
    #   总和 / 类别个数 = 均值
    #---------------------------------------------#
    return dice / x.shape[1]

#---------------------------------------------#
#   Dice loss (objective to minimize) between 0 and 1
#---------------------------------------------#
def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    #---------------------------------------------#
    #   得到像素针对每个类别的概率
    #---------------------------------------------#
    x = nn.functional.softmax(x, dim=1)
    #---------------------------------------------#
    #   multiclass_dice_coeff: 分别计算每个类别的损失
    #---------------------------------------------#
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    #---------------------------------------------#
    #   计算diceloss
    #---------------------------------------------#
    return 1 - fn(x, target, ignore_index=ignore_index)
