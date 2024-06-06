import os
from enum import Enum
import glob
import PIL
import torch
from torchvision import transforms
import random
import numpy as np
_CLASSNAMES = [
 "bagel","cable_gland","carrot","cookie","dowel","foam","peach","potato","rope","tire"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"


class GeneratedMVTec3DDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        ori_source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        selected_anomaly_class=None,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        seed = kwargs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.source = source
        self.selected_anomaly_class = selected_anomaly_class
        self.split = split
        self.ori_source = ori_source
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
 
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly_class, image_path, mask_path, ori_img_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        ori_img = PIL.Image.open(ori_img_path).convert("RGB")
        ori_img = self.transform_img(ori_img)
        # if self.split == DatasetSplit.TEST and mask_path is not None:
        mask = PIL.Image.open(mask_path)
        mask = self.transform_mask(mask)
        # else:
        #     mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "ori_img":ori_img,
            "mask": mask,
            "classname": classname,
            "anomaly_class": anomaly_class,
            "image_name": "/".join(image_path.split("/")[-3:]),
            "image_path": image_path,
            "ori_img_path": ori_img_path
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname)
            ori_img_classpath = os.path.join(self.ori_source, classname+"/train/good/")
            ori_imgs = sorted(glob.glob(ori_img_classpath+"rgb/*.png"))
            maskpath = os.path.join(self.source+"_mask", classname)
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in self.selected_anomaly_class:  # 异常种类
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [  # imgpaths_per_class为双重字典 ，第二级别字典的value是一个文件路径的列表
                    os.path.join(anomaly_path, x) for x in anomaly_files  # os.listdir必须这样 
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                anomaly_mask_path = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                ]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):  # 多层级字典 metal_nut good /metal_nut/01.png
                    if not os.path.isdir(image_path):
                        data_tuple = [classname, anomaly, image_path]  # metal_nut banded /metal_nut/banded/01.png
                        ori_idx = int(image_path.split("_")[-2].split("/")[-1])
                        ori_img_path = ori_imgs[ori_idx]
                        mask_path = maskpaths_per_class[classname][anomaly][i]
                        data_tuple.append(mask_path)
                        data_tuple.append(ori_img_path)
                        data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate  #不成组返回图片
    


class GeneratedMVTec3DDatasetForRelationNet(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        ori_source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        selected_anomaly_class=None,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        seed = kwargs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.source = source
        self.selected_anomaly_class = selected_anomaly_class
        self.split = split
        self.ori_source = ori_source
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
 
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        my_list = self.data_to_iterate[idx]
        res_list = []
        for i in range(len(my_list)):
            classname, anomaly_class, image_path, mask_path, ori_img_path = my_list[i]
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.transform_img(image)
            ori_img = PIL.Image.open(ori_img_path).convert("RGB")
            ori_img = self.transform_img(ori_img)
            # if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
            item = {
            "image": image,
            "ori_img":ori_img,
            "mask": mask,
            "classname": classname,
            "anomaly_class": anomaly_class,
            "image_name": "/".join(image_path.split("/")[-3:]),
            "image_path": image_path,
            "ori_img_path": ori_img_path
        }
            res_list.append(item)
        # else:
        #     mask = torch.zeros([1, *image.size()[1:]])

        return res_list

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname)
            ori_img_classpath = os.path.join(self.ori_source, classname+"/train/good/")
            ori_imgs = sorted(glob.glob(ori_img_classpath+"rgb/*.png"))
            maskpath = os.path.join(self.source+"_mask", classname)
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in self.selected_anomaly_class:  # 异常种类
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [  # imgpaths_per_class为双重字典 ，第二级别字典的value是一个文件路径的列表
                    os.path.join(anomaly_path, x) for x in anomaly_files  # os.listdir必须这样 
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                anomaly_mask_path = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                ]

        # Unrolls the data dictionary to an easy-to-iterate list.
        
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for i in range(len(list(imgpaths_per_class[classname].values())[0])):
                _ = list(imgpaths_per_class[classname].keys())
                data_item = []
                for anomaly in sorted(_):
                    image_path = imgpaths_per_class[classname][anomaly][i]  # 多层级字典 metal_nut good /metal_nut/01.png
                    if not os.path.isdir(image_path):
                        data_tuple = [classname, anomaly, image_path]  # metal_nut banded /metal_nut/banded/01.png
                        ori_idx = int(image_path.split("_")[-2].split("/")[-1])
                        ori_img_path = ori_imgs[ori_idx]
                        mask_path = maskpaths_per_class[classname][anomaly][i]
                        data_tuple.append(mask_path)
                        data_tuple.append(ori_img_path)
                        data_item.append(data_tuple)
                data_to_iterate.append(data_item)
        return imgpaths_per_class, data_to_iterate  # 和上面那个不同，这个每次返回一组缺陷图（比如4个）
