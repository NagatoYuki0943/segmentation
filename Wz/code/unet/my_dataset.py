import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        """
        root: ./DRIVE
        train: train or test
        transforms: 数据预处理方式
        """
        super(DriveDataset, self).__init__()
        #---------------------------------------------#
        #   拼接路径
        #---------------------------------------------#
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        #---------------------------------------------#
        #   找image下所有图片名称
        #---------------------------------------------#
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        #---------------------------------------------#
        #   得到图片路径
        #---------------------------------------------#
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        #---------------------------------------------#
        #   得到manual路径,是真实标签,血管是255,背景为0
        #---------------------------------------------#
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        #---------------------------------------------#
        #   检查manual文件是否存在
        #---------------------------------------------#
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        #---------------------------------------------#
        #   得到mask路径,mask是感兴趣区域的遮罩,感兴趣区域为255,不感兴趣区域为0
        #---------------------------------------------#
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        #---------------------------------------------#
        #   检查mask文件是否存在
        #---------------------------------------------#
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img      = Image.open(self.img_list[idx]).convert('RGB')
        manual   = Image.open(self.manual[idx]).convert('L')
        manual   = np.array(manual) / 255                           # manual归一化,血管是255,归一化变为1,背景为0
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')      # mask 感兴趣区域为255,不感兴趣区域为0
        roi_mask = 255 - np.array(roi_mask)                         # 522 - mask 感兴趣区域为0,不感兴趣区域为255,计算损失时忽略255即可
        mask     = np.clip(manual + roi_mask, a_min=0, a_max=255)   # manual+mask  背景为0 血管为1,不感兴趣为255

        #---------------------------------------------#
        #   这里转回PIL的原因是，transforms中是对PIL数据进行处理
        #---------------------------------------------#
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        #---------------------------------------------#
        #   img:  图片
        #   mask: manual+mask  背景为0 血管为1,不感兴趣为255
        #---------------------------------------------#
        return img, mask

    def __len__(self):
        return len(self.img_list)

    #---------------------------------------------#
    #   打包方法
    #---------------------------------------------#
    @staticmethod
    def collate_fn(batch):
        """
        batch: [[image1, target1],[image2, target2]...]
        """
        # [image1, image2...] [target1, target2...]
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


#---------------------------------------------#
#   将同一批数据调整为相同大小,训练集默认尺寸相同,验证不同
#---------------------------------------------#
def cat_list(images, fill_value=0):
    #---------------------------------------------#
    #   计算该batch数据中，channel, h, w的最大值
    #---------------------------------------------#
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))   # 假设 max_size: (3. 512, 512)
    batch_shape = (len(images),) + max_size                                 # 增加batch      (4, 3, 512, 512)
    #---------------------------------------------#
    #   new重新构建tensor,形状是最大的形状,用0填充,然后放入图片
    #---------------------------------------------#
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

