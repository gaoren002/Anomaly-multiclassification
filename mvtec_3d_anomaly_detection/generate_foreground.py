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
    to_be_processed = ["foam"]#,"tire"]
    for i, item in enumerate(to_be_processed):
        os.chdir(item)
        os.makedirs("./train/good/fore_mask", exist_ok=True)
        file_paths = glob.glob("./train/good/rgb/*.png")
        for j, file_path in enumerate(file_paths):
            sequence = file_path.split(".")[-2].split("/")[-1]
            image = cv2.imread(file_path)
            mask = np.zeros(image.shape[:2], dtype="uint8")
            rect = (100, 100, image.shape[1]-150, image.shape[0]-150)
            fgModel = np.zeros((1, 65), dtype="float")
            bgModel = np.zeros((1, 65), dtype="float")
            # apply GrabCut using the the bounding box segmentation method
            start = time.time()
            (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel, fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
            valueMask = (mask == cv2.GC_PR_FGD).astype("uint8") * 255
            cv2.imwrite(f"./train/good/fore_mask/mask_{sequence}.png", valueMask)
            end = time.time()
            print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))
        os.chdir("../")