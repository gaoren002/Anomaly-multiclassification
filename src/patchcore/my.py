import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)

def predict(self, data):
    if isinstance(data, torch.utils.data.DataLoader):
        return self._predict_dataloader(data)
    return self._predict(data)


def _predict_dataloader(self, dataloader):
    """This function provides anomaly scores/maps for full dataloaders."""
    _ = self.forward_modules.eval()

    scores = []
    masks = []
    labels_gt = []
    masks_gt = []
    with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:

        for image in data_iterator:
            if isinstance(image, dict):
                labels_gt.extend(image["is_anomaly"].numpy().tolist())
                masks_gt.extend(image["mask"].numpy().tolist())
                image = image["image"]
            _scores, _masks = self._predict(image)
            for score, mask in zip(_scores, _masks):
                scores.append(score)
                masks.append(mask)
    return scores, masks, labels_gt, masks_gt


def _predict(self, images):
    """Infer score and mask for a batch of images."""
    images = images.to(torch.float).to(self.device)
    _ = self.forward_modules.eval()

    batchsize = images.shape[0]
    with torch.no_grad():
        features, patch_shapes = self._embed(images, provide_patch_shapes=True)
        features = np.asarray(features)

        patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
        image_scores = self.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = self.patch_maker.score(image_scores)

        patch_scores = self.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        )
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

        masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

    return [score for score in image_scores], [mask for mask in masks]