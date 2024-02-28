import os

import torch.utils.data as data
from PIL import Image


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        """
        transforms: 数据预处理方法
        txt_name:   读取哪一个txt
        """
        super(VOCSegmentation, self).__init__()
        #---------------------------------------------#
        #   VOC路径
        #---------------------------------------------#
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        #---------------------------------------------#
        #   读取txt得到文件名
        #---------------------------------------------#
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        #---------------------------------------------#
        #   删除换行符和空行
        #---------------------------------------------#
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        #---------------------------------------------#
        #   根据文件名找到jpg和png图片
        #---------------------------------------------#
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        #---------------------------------------------#
        #   分别找到image和target并通过预处理
        #---------------------------------------------#
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

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
#   将同一批数据调整为相同大小,训练集默认尺寸相同,验证不同,所以需要这个方法
#---------------------------------------------#
def cat_list(images, fill_value=0):
    #---------------------------------------------#
    #   计算该batch数据中，channel, h, w的最大值
    #---------------------------------------------#
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))   # 假设 max_size: (3. 480, 480)
    batch_shape = (len(images),) + max_size                                 # 增加batch      (4, 3, 480, 480)
    #---------------------------------------------#
    #   new重新构建tensor,形状是最大的形状,用0填充,然后放入图片
    #---------------------------------------------#
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
