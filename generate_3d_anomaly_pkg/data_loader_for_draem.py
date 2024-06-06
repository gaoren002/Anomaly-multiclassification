import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

# class MVTecDRAEMTestDataset(Dataset):

#     def __init__(self, root_dir, resize_shape=None):
#         self.root_dir = root_dir
#         self.images = sorted(glob.glob(root_dir+"/*/*.png"))
#         self.resize_shape=resize_shape

#     def __len__(self):
#         return len(self.images)

#     def transform_image(self, image_path, mask_path):
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         if mask_path is not None:
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         else:
#             mask = np.zeros((image.shape[0],image.shape[1]))
#         if self.resize_shape != None:
#             image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
#             mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

#         image = image / 255.0
#         mask = mask / 255.0

#         image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
#         mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

#         image = np.transpose(image, (2, 0, 1))
#         mask = np.transpose(mask, (2, 0, 1))
#         return image, mask

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_path = self.images[idx]
#         dir_path, file_name = os.path.split(img_path)
#         base_dir = os.path.basename(dir_path)
#         if base_dir == 'good':
#             image, mask = self.transform_image(img_path, None)
#             has_anomaly = np.array([0], dtype=np.float32)
#         else:
#             mask_path = os.path.join(dir_path, '../../ground_truth/')
#             mask_path = os.path.join(mask_path, base_dir)
#             mask_file_name = file_name.split(".")[0]+"_mask.png"
#             mask_path = os.path.join(mask_path, mask_file_name)
#             image, mask = self.transform_image(img_path, mask_path)
#             has_anomaly = np.array([1], dtype=np.float32)

#         sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

#         return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None,generate_times = 1):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.anomaly_list=["banded",
                       "blotchy",
                       "braided",
                       "bubbly",
                       "bumpy",
                       "chequered",
                       "cobwebbed",
                       "cracked",
                       "crosshatched",
                       "crystalline",
                       "dotted",
                       "fibrous",
                       "flecked",
                       "freckled",
                       "frilly",
                       "gauzy",
                       "grid",
                       "grooved",
                       "honeycombed",
                       "interlaced",
                       "knitted",
                       "lacelike",
                       "lined",
                       "marbled",
                       "matted",
                       "meshed",
                       "paisley",
                       "perforated",
                       "pitted",
                       "pleated",
                       "polka-dotted",
                       "porous",
                       "potholed",
                       "scaly",
                       "smeared",
                       "spiralled",
                       "sprinkled",
                       "stained",
                       "stratified",
                       "striped",
                       "studded",
                       "swirly",
                       "veined",
                       "waffled",
                       "woven",
                       "wrinkled",
                       "zigzagged"
                       ]
        self.generate_times = generate_times
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.fore_mask = False
        self.image_paths = sorted(glob.glob(root_dir+"/rgb/*.png"))
        self.pre_mask_paths = None
        if os.path.exists(root_dir + "/fore_mask"):
            self.pre_mask_paths = sorted(glob.glob(root_dir+"/fore_mask/*.png"))
        self.anomaly_source_paths_classdir = sorted(os.listdir(anomaly_source_path))
        self.anomaly_source_paths = {item:sorted(glob.glob(anomaly_source_path+"/"+item+"/*.jpg")) for item in  self.anomaly_source_paths_classdir} # 字典{缺陷类别：对应文件路径列表}
        anomaly_class_paths_number = [len(self.anomaly_source_paths[self.anomaly_list[i]]) for i in range(len(self.anomaly_source_paths_classdir))]
        self.anomaly_class_paths_number_min = min(anomaly_class_paths_number)
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        

    def __len__(self):
        return len(self.image_paths)*len(self.anomaly_source_paths_classdir)*self.generate_times # self.anomaly_class_paths_number_min  #220*47*次数或者每一类缺陷原图下的图片数量的最小值


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, pre_mask, anomaly_source_path):
        area = 0.1
        while area <=0.1 or area>=0.4:
            aug = self.randAugmenter()
            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = cv2.imread(anomaly_source_path)
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

            anomaly_img_augmented = aug(image=anomaly_source_img)
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            # perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.35
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
            area = np.sum(perlin_thr) / (self.resize_shape[0]*self.resize_shape[1])
            perlin_thr_new = perlin_thr *pre_mask[...,None] if pre_mask is not None else perlin_thr
            img_thr = anomaly_source_img * perlin_thr_new / 255.0
        
            beta = np.random.rand(1) / 2

            augmented_image = image * (1 - perlin_thr_new) + (1 - beta) * img_thr + beta * image * perlin_thr_new

            no_anomaly = 0
            
        
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr_new, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            msk_masked = (perlin_thr_new).astype(np.float32)
            augmented_image = msk_masked * augmented_image + (1-msk_masked)*image  # 貌似无用
            has_anomaly = 1.0
            if np.sum(msk_masked) == 0:
                has_anomaly=0.0
            return augmented_image, msk, msk_masked, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_paths, pre_mask_paths, idx, anomaly_source_path):
        image = cv2.imread(image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_class = anomaly_source_path.split("/")[-2]
        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)
        if pre_mask_paths is not None:
            pre_mask = cv2.imread(pre_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            pre_mask = cv2.resize(pre_mask, dsize=(self.resize_shape[1], self.resize_shape[0]))
            pre_mask = pre_mask / 255.0
        else:
            pre_mask = None
        image = image / 255.0 # np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask_ori, anomaly_mask, has_anomaly = self.augment_image(image, pre_mask, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        anomaly_mask_ori = np.transpose(anomaly_mask_ori, (2, 0, 1))
        return image, augmented_image, anomaly_mask_ori, anomaly_mask, has_anomaly, anomaly_class

    def __getitem__(self, idx):
        
        anomaly_class_idx = idx % len(self.anomaly_source_paths_classdir) #即47
        anomaly_class_paths = self.anomaly_source_paths[self.anomaly_list[anomaly_class_idx]]
        idx = idx // len(self.anomaly_source_paths_classdir)
        
        ori_img_index = idx % len(self.image_paths)
        idx = idx // len(self.image_paths)
        
        anomaly_source_idx_in_class = idx % len(anomaly_class_paths)
        image, augmented_image, anomaly_mask_ori, anomaly_mask, has_anomaly, anomaly_class = self.transform_image(self.image_paths, self.pre_mask_paths,ori_img_index,
                                                                           anomaly_class_paths[anomaly_source_idx_in_class])
        sample = {'image': image, "anomaly_mask": anomaly_mask,"anomaly_mask_ori": anomaly_mask_ori,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': ori_img_index, 'anomaly_source_idx':anomaly_source_idx_in_class, "anomaly_class":anomaly_class}

        return sample
