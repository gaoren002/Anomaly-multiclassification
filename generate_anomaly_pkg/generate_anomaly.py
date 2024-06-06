import torch
import os
import sys
a = sys.path
from data_loader_for_draem import MVTecDRAEMTrainDataset
b = sys.path
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
import os
import numpy as np
from PIL import Image
import cv2
data_path = "./mvtec_anomaly_detection"
anomaly_source_path = "./generate_anomaly_pkg/dtd/images"
obj_list = [#'capsule',
            #'bottle',
            #'carpet',
            #'leather',
            #'pill',
            #'transistor',
            # 'tile',
            # 'cable',
            # 'zipper',
            # 'toothbrush',
            # 'metal_nut',
            # 'hazelnut',
            # 'screw',
            # 'grid',
            # 'wood',
            "miniLED"
            ]
if not os.path.exists("./generate_anomaly_pkg/sythesized_anomaly_mask_ori/"):
        os.mkdir("./generate_anomaly_pkg/sythesized_anomaly_mask_ori/")
if not os.path.exists("./generate_anomaly_pkg/sythesized_anomaly_mask/"):
    os.mkdir("./generate_anomaly_pkg/sythesized_anomaly_mask/")
if not os.path.exists("./generate_anomaly_pkg/sythesized_anomaly/"):
    os.mkdir("./generate_anomaly_pkg/sythesized_anomaly/")
for obj_name in obj_list:
    data_idx = -1
    dataset = MVTecDRAEMTrainDataset(data_path + "/"+obj_name + "/train/good", anomaly_source_path, resize_shape=[256, 256],generate_times=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    if not  os.path.exists("./generate_anomaly_pkg/sythesized_anomaly/"+obj_name):
        os.mkdir("./generate_anomaly_pkg/sythesized_anomaly/"+obj_name)
    if not  os.path.exists("./generate_anomaly_pkg/sythesized_anomaly_mask/"+obj_name):
        os.mkdir("./generate_anomaly_pkg/sythesized_anomaly_mask/"+obj_name)
    if not  os.path.exists("./generate_anomaly_pkg/sythesized_anomaly_mask_ori/"+obj_name):
        os.mkdir("./generate_anomaly_pkg/sythesized_anomaly_mask_ori/"+obj_name)
    for sample in dataloader:
        length = len(dataloader)
       
        if os.path.exists("./generate_anomaly_pkg/sythesized_anomaly/"+obj_name+"/"+sample['anomaly_class'][0])==False:
            os.mkdir("./generate_anomaly_pkg/sythesized_anomaly/"+obj_name+"/"+sample['anomaly_class'][0])
        if os.path.exists("./generate_anomaly_pkg/sythesized_anomaly_mask/"+obj_name+"/"+sample['anomaly_class'][0])==False:
            os.mkdir("./generate_anomaly_pkg/sythesized_anomaly_mask/"+obj_name+"/"+sample['anomaly_class'][0])
        if os.path.exists("./generate_anomaly_pkg/sythesized_anomaly_mask_ori/"+obj_name+"/"+sample['anomaly_class'][0])==False:
            os.mkdir("./generate_anomaly_pkg/sythesized_anomaly_mask_ori/"+obj_name+"/"+sample['anomaly_class'][0])
        Image.fromarray((sample['augmented_image'].numpy()[0].transpose(1,2,0)*255.0).astype(np.uint8)).save("./generate_anomaly_pkg/sythesized_anomaly/"+obj_name+"/"+
                                                                                                                sample['anomaly_class'][0]+
                                                                                                                "/"+str(sample["idx"].numpy()[0])+
                                                                                                                "_"+str(sample["anomaly_source_idx"].numpy()[0])+
                                                                                                                ".png")
        mymask = (sample['anomaly_mask'].numpy()[0].transpose(1,2,0)*255.0).astype(np.uint8)
        mymask_name = "./generate_anomaly_pkg/sythesized_anomaly_mask/"+obj_name+"/"+\
                        sample['anomaly_class'][0]+\
                        "/"+str(sample["idx"].numpy()[0])+\
                        "_"+str(sample["anomaly_source_idx"].numpy()[0])+\
                        "_mask.png"
        if not os.path.exists(mymask_name):
            cv2.imwrite(mymask_name, mymask)

        mymask_ori = (sample['anomaly_mask_ori'].numpy()[0].transpose(1,2,0)*255.0).astype(np.uint8)
        mymask_ori_name = "./generate_anomaly_pkg/sythesized_anomaly_mask_ori/"+obj_name+"/"+\
                    sample['anomaly_class'][0]+\
                    "/"+str(sample["idx"].numpy()[0])+\
                    "_"+str(sample["anomaly_source_idx"].numpy()[0])+\
                    "_mask_ori.png"
        if not os.path.exists(mymask_ori_name):
            cv2.imwrite(mymask_ori_name, mymask_ori)