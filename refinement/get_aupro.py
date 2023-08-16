import numpy as np
from skimage import measure
from sklearn.metrics import auc

def run(score_imgs, labeled_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)
    score_imgs = np.array(score_imgs)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())