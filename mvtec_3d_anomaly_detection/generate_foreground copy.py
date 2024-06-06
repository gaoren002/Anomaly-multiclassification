import cv2
import time
import numpy as np
import glob
import os

if __name__ == "__main__":
    paths = os.listdir("./mvtec_3d_anomaly_detection")
    dirs = list()
    os.chdir("./mvtec_3d_anomaly_detection")
    for i in range(len(paths)):
        if os.path.isdir(paths[i]):
            dirs.append(paths[i])
    to_be_processed = ["bagel","cable_gland","carrot","cookie","dowel","peach","potato"]
    for i, item in enumerate(to_be_processed):
        if item is not "metal_nut":
            os.chdir(item)
            os.makedirs("./train/good/fore_mask", exist_ok=True)
            file_paths = glob.glob("./train/good/rgb/*.png")
            for j, file_path in enumerate(file_paths):
                sequence = file_path.split(".")[-2].split("/")[-1]
                image = cv2.imread(file_path)
                mask1 = np.zeros([image.shape[0]+2, image.shape[1]+2], dtype="uint8")
                mask2 = np.zeros([image.shape[0]+2, image.shape[1]+2], dtype="uint8")
                rect = (0, 0, image.shape[1]-1, image.shape[0]-1)

                # apply GrabCut using the the bounding box segmentation method
                start = time.time()
                cv2.floodFill(image, mask1, (0, 0), (0, 0, 0), (80, 80, 80), (50, 50, 50), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255<<8))
                cv2.floodFill(image, mask2, (image.shape[1]-1, image.shape[0]-1), (0, 0, 0), (80, 80, 80), (50, 50, 50), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255<<8))
                mask = 255 - (mask1 | mask2)
                cv2.imwrite(f"./train/good/fore_mask/mask_{sequence}.png", mask)
                end = time.time()
                print("[INFO] applying floodfill took {:.5f} seconds".format(end - start))
            os.chdir("../")
        else:
            os.chdir(item)
            os.makedirs("./train/good/fore_mask", exist_ok=True)
            file_paths = glob.glob("./train/good/rgb/*.png")
            for j, file_path in enumerate(file_paths):
                sequence = file_path.split(".")[-2].split("/")[-1]
                image = cv2.imread(file_path)
                mask1 = np.zeros([image.shape[0]+2, image.shape[1]+2], dtype="uint8")
                mask2 = np.zeros([image.shape[0]+2, image.shape[1]+2], dtype="uint8")
                mask3 = np.zeros([image.shape[0]+2, image.shape[1]+2], dtype="uint8")
                rect = (0, 0, image.shape[1]-1, image.shape[0]-1)

                # apply GrabCut using the the bounding box segmentation method
                start = time.time()
                cv2.floodFill(image, mask1, (0, 0), (0, 0, 0), (80, 80, 80), (50, 50, 50), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255<<8))
                cv2.floodFill(image, mask2, (image.shape[1]-1, image.shape[0]-1), (0, 0, 0), (80, 80, 80), (50, 50, 50), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255<<8))
                cv2.floodFill(image, mask3, ((image.shape[1]-1)//2, (image.shape[0]-1)//2), (0, 0, 0), (80, 80, 80), (30, 30, 30), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255<<8))
                mask = 255 - (mask1 | mask2 | mask3)
                cv2.imwrite(f"./train/good/fore_mask/mask_{sequence}.png", mask)
                end = time.time()
                print("[INFO] applying floodfill took {:.5f} seconds".format(end - start))
            os.chdir("../")