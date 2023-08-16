import numpy as np
from refinement.utils import metrics
import logging
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

def run(PRED_MAX, GT_MAX, PRED, GT, get_pred=False, get_GT=False):

    PRED_MAX = np.expand_dims(np.nan_to_num(np.array(PRED_MAX)), axis=0)
    min_PRED_MAX = PRED_MAX.min(axis=-1).reshape(-1, 1)
    max_PRED_MAX = PRED_MAX.max(axis=-1).reshape(-1, 1)
    PRED_MAX = (PRED_MAX - min_PRED_MAX) / (max_PRED_MAX - min_PRED_MAX)
    PRED_MAX = np.mean(PRED_MAX, axis=0)

    GT_MAX = np.expand_dims(np.nan_to_num(np.array(GT_MAX)), axis=0)
    min_GT_MAX = GT_MAX.min(axis=-1).reshape(-1, 1)
    max_GT_MAX = GT_MAX.max(axis=-1).reshape(-1, 1)
    GT_MAX = (GT_MAX - min_GT_MAX) / (max_GT_MAX - min_GT_MAX)
    GT_MAX = np.mean(GT_MAX, axis=0)

    PRED = np.expand_dims(np.nan_to_num(np.array(PRED)), axis=0)
    min_PRED = (
        PRED.reshape(len(PRED), -1)
        .min(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    max_PRED = (
        PRED.reshape(len(PRED), -1)
        .max(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    PRED = (PRED - min_PRED) / (max_PRED - min_PRED)
    PRED = np.mean(PRED, axis=0)

    GT = np.expand_dims(np.nan_to_num(np.array(GT)), axis=0)
    min_GT = (
        GT.reshape(len(GT), -1)
        .min(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    max_GT = (
        GT.reshape(len(GT), -1)
        .max(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    GT = (GT - min_GT) / (max_GT - min_GT)
    GT = np.mean(GT, axis=0)

    LOGGER.info("Computing evaluation metrics.")
    image_scores = metrics.compute_imagewise_retrieval_metrics(
        PRED_MAX, GT_MAX
    )
    image_auroc = image_scores["auroc"]
    #print("image_scores")

    # Compute PRO score & PW Auroc for all images
    if GT.shape[1] > 64:
        image_size = [64, 64]
        interpolate_bilinear = nn.Upsample(size=image_size, mode='bilinear')
        GT = interpolate_bilinear(torch.unsqueeze(torch.asarray(GT), 1)).squeeze().numpy()
        PRED = interpolate_bilinear(torch.unsqueeze(torch.asarray(PRED), 1)).squeeze().numpy()

    pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
        PRED, GT
    )
    pixel_auroc = pixel_scores["auroc"]
    #print("pixel_scores")

    thresholds = np.expand_dims(image_scores["thresholds"], axis=1)
    image_TP, image_FP, image_FN, image_TN = np.expand_dims(image_scores["TP"], axis=1), np.expand_dims(image_scores["FP"], axis=1), np.expand_dims(image_scores["FN"], axis=1), np.expand_dims(image_scores["TN"], axis=1)
    image_precision, image_recall, image_F1 = np.expand_dims(image_scores["precision"], axis=1), np.expand_dims(image_scores["recall"], axis=1), np.expand_dims(image_scores["F1"], axis=1)
    pixel_TP, pixel_FP, pixel_FN, pixel_TN = np.expand_dims(pixel_scores["TP"], axis=1), np.expand_dims(pixel_scores["FP"], axis=1), np.expand_dims(pixel_scores["FN"], axis=1), np.expand_dims(pixel_scores["TN"], axis=1)
    pixel_precision, pixel_recall, pixel_F1 = np.expand_dims(pixel_scores["precision"], axis=1), np.expand_dims(pixel_scores["recall"], axis=1), np.expand_dims(pixel_scores["F1"], axis=1)
    all_scores = np.concatenate(
        (
            thresholds,
            image_TP, image_FP, image_FN, image_TN,
            image_precision, image_recall, image_F1,
            pixel_TP, pixel_FP, pixel_FN, pixel_TN,
            pixel_precision, pixel_recall, pixel_F1,
        )
        , axis=1)

    if get_pred == False:
        return all_scores, image_auroc, pixel_auroc
    elif get_pred == True:
        image_pred = PRED_MAX
        pixel_pred = PRED
        image_min, image_max, pixel_min, pixel_max = min_PRED_MAX, max_PRED_MAX, min_PRED, max_PRED
        if get_GT == False:
            return all_scores, image_auroc, pixel_auroc, image_pred, image_min, image_max, pixel_pred, pixel_min, pixel_max
        elif get_GT == True:
            image_GT = GT_MAX
            pixel_GT = GT
            return all_scores, image_auroc, pixel_auroc, image_pred, image_min, image_max, pixel_pred, pixel_min, pixel_max, image_GT, pixel_GT

def main(**kwargs):
    run()

if __name__ == "__main__":
    main()