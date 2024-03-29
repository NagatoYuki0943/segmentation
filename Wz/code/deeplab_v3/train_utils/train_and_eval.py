import torch
from torch import nn
import train_utils.distributed_utils as utils


#---------------------------------------------#
#   计算损失
#---------------------------------------------#
def criterion(inputs, target):
    """
    input: {'out': Tensor, 'aux': Tensor}
    """
    losses = {}
    #---------------------------------------------#
    #   遍历主分支和aux分支
    #---------------------------------------------#
    for name, x in inputs.items():
        #---------------------------------------------#
        #   忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        #---------------------------------------------#
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    #---------------------------------------------#
    #   不使用辅助分支
    #---------------------------------------------#
    if len(losses) == 1:
        return losses['out']

    #---------------------------------------------#
    #   使用辅助分支
    #---------------------------------------------#
    return losses['out'] + 0.5 * losses['aux']


#---------------------------------------------#
#   验证
#---------------------------------------------#
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    #---------------------------------------------#
    #   混淆矩阵
    #---------------------------------------------#
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            #---------------------------------------------#
            #   output: {'out': Tensor, 'aux': Tensor}
            #---------------------------------------------#
            output = model(image)
            output = output['out']  # 只使用主分支输出

            #---------------------------------------------#
            #   结果和target展平放入混淆矩阵
            #   结果放入每个的最大类别 argmax(1) 找channel最大值
            #---------------------------------------------#
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    #---------------------------------------------#
    #   循环数据
    #---------------------------------------------#
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            #---------------------------------------------#
            #   output: {'out': Tensor, 'aux': Tensor}
            #---------------------------------------------#
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        #---------------------------------------------#
        #   注意这里每更新一个step就更新学习率,而不是一个epoch
        #---------------------------------------------#
        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    #---------------------------------------------#
    #   返回损失和学习率
    #---------------------------------------------#
    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,          # 训练一个epoch要走多少step
                        epochs: int,            # 训练epoch数
                        warmup=True,
                        warmup_epochs=1,        # 热身训练要保持1轮
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        x: step
        return:
            倍率因子
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            #---------------------------------------------------------#
            #   warmup过程中lr倍率因子从warmup_factor -> 1
            #---------------------------------------------------------#
            return warmup_factor * (1 - alpha) + alpha
        else:
            #---------------------------------------------------------#
            #   warmup后lr倍率因子从1 -> 0
            #   参考deeplab_v2: Learning rate policy
            #   注意减去了$warmup_epochs * num_step$,它是warp总step数
            #   lr \times (1 - \frac {iter} {max\_iter})^{power}
            #---------------------------------------------------------#
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
