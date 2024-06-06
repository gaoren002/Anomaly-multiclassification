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
MA_path =  "./generate_anomaly_pkg/sythesized_anomaly_mask/metal_nut/banded/20_0_mask.png"# "./generate_anomaly_pkg/dtd/images/banded/banded_0002.jpg"
O_path = "./mvtec_anomaly_detection/metal_nut/train/good/000.png"
A_path = "./generate_anomaly_pkg/dtd/images/banded/banded_0002.jpg"
MAA_path = "./cv21.jpg"
MA_image = cv2.imread(MA_path, cv2.IMREAD_COLOR)
height_color, width_color, channels_color = MA_image.shape
MAA_image = cv2.imread(MAA_path, cv2.IMREAD_COLOR)
MAA_image = cv2.resize(MAA_image,(width_color,height_color))
O_image = cv2.imread(O_path, cv2.IMREAD_COLOR)
O_image = cv2.resize(O_image,(width_color,height_color))
A_image = cv2.imread(A_path, cv2.IMREAD_COLOR)
A_image = cv2.resize(A_image,(width_color,height_color))
A_image = cv2.cvtColor(A_image,cv2.COLOR_BGR2RGB)
A_image =  (MAA_image/255.0)*(O_image/255.0)*255.0


cv2.imwrite('cv21111.jpg', A_image)
