import pytorch_lightning as pl
import argparse
import os
from pathlib import Path
import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--project_root_path', default=r'./result/refine_result/ensemble_optimize_refine')
    parser.add_argument('--ensemble_root_path', default=r'./result/refine_result/ensemble_optimize_refine/viz')
    
    args = parser.parse_args()
    return args

def get_ensemble_pxl_score(score_np) :
    scores_ravel = score_np.reshape(score_np.shape[0], -1)
    min_scores = scores_ravel.min(axis=-1).reshape(-1, 1, 1, 1)
    max_scores = scores_ravel.max(axis=-1).reshape(-1, 1, 1, 1)
    score_np = (score_np - min_scores) / (max_scores - min_scores)
    scores = np.mean(score_np, axis=0)
    
    return scores

def get_ensemble_img_score(score_np, anomaly_percentage = 0.001) :
    backbone_num, img_num, H, W = score_np.shape
    scores_ravel = score_np.reshape(backbone_num * img_num, -1)
    
    # mean_num = (int)(H * W * anomaly_percentage)
    
    # scores_ravel_sorted = -np.sort(-scores_ravel, axis = 1)
    # img_score = np.mean(scores_ravel_sorted[:, :mean_num], axis = 1).reshape(backbone_num, img_num)
    # img_score = np.mean(img_score, axis = 0)
    
    max_scores = scores_ravel.max(axis=1).reshape(backbone_num, img_num)
    img_score = np.mean(max_scores, axis = 0)
    
    return img_score

def min_max_norm(image, thres=-1):
    a_min, a_max = image.min(), image.max()
    if thres == -1 :
        return (image - a_min)/(a_max - a_min)
    else :
        soft_thres = thres / 2
        restricted = np.maximum((image - soft_thres)/(a_max - soft_thres), 0)
        return restricted
    
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def image_with_ground_truth(img, gt, kernel_size = (11, 11)):
    kernel = np.ones(kernel_size, np.uint8)
    gt = np.float32(gt)
    erode_gt = cv2.erode(gt, kernel)
    edge_gt = gt - erode_gt
    
    red_edge_gt = np.zeros(shape=(edge_gt.shape + (3,)))
    red_edge_gt[:, :, 1] = 128 * edge_gt
    red_edge_gt[:, :, 2] = 255 * edge_gt
    
    img_part_mask = np.zeros_like(red_edge_gt)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_gt < 0.1)
    
    out = red_edge_gt + img * img_part_mask
    return np.uint8(out)
        
if __name__ == '__main__':
    args = get_args()
    print(f"Start calculating anomaly score!")
    ensemble_result_dir = Path(args.ensemble_root_path)
    ensemble_result_dir.mkdir(parents=True, exist_ok=True)
    
    category_border = [0, 83, 233, 365, 482, 560, 670, 794, 909, 1076, 1236, 1353, 1395, 1495, 1574, 1725]
    category_list = ["Bottle", "Cable", "Capsule", "Carpet", "Grid", "Hazelnut", "Leather", "Metal_nut", "Pill", "Screw", "Tile", "Toothbrush", "Transistor", "Wood", "Zipper"]
    gt_list = []
    patchcore_gt_list = []
    amap_list = [] 
    img_list = []
    resized_img_list = []
    patchcore_amap_list = []  
    gt_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "*_gt.png")))
    amap_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "*_pred.png")))
    img_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "image*.png")))
    patchcore_amap_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "patchcore", "*.png")))

    pxl_score_list = []
    img_score_list = []
    threshold_list = []
    patchcore_threshold_list = []
    
    for idx, gt_path in enumerate(gt_path_list) :
        amap_path = amap_path_list[idx]
        img_path = img_path_list[idx]
        patchcore_amap_path = patchcore_amap_path_list[idx]
        gt = (np.array(Image.open(gt_path)) / 65535).astype(int) 
        
        gt_list.append((np.array(Image.open(gt_path)) / 65535).astype(int))
        amap = np.array(Image.open(amap_path)) / 65535
        amap_list.append(amap)
        patchcore_amap = np.array(Image.open(patchcore_amap_path)) / 65535
        patchcore_amap_list.append(patchcore_amap)
        
        resized_gt_shape = (240, 240)
        resized_gt = np.uint8(cv2.resize(np.float32(gt), resized_gt_shape, interpolation=cv2.INTER_NEAREST))
        patchcore_amap_shape = patchcore_amap.shape[:2]        
        cropped_gt = resized_gt[resized_gt_shape[0]//2 - patchcore_amap_shape[0]//2:resized_gt_shape[0]//2 + patchcore_amap_shape[0]//2, \
            resized_gt_shape[1]//2 - patchcore_amap_shape[1]//2:resized_gt_shape[1]//2 + patchcore_amap_shape[1]//2]
        patchcore_gt_list.append(cropped_gt)
            
        origin_image = np.array(Image.open(img_path))
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
        if len(origin_image.shape) == 2:
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_GRAY2BGR)
        
        amap_shape = amap.shape[:2]
        origin_shape = origin_image.shape[:2]
        cropped_image = origin_image[origin_shape[0]//2 - amap_shape[0]//2:origin_shape[0]//2 + amap_shape[0]//2, \
            origin_shape[1]//2 - amap_shape[1]//2:origin_shape[1]//2 + amap_shape[1]//2]
        
        img_list.append(cropped_image)
        
        resized_img_shape = (240, 240)
        resized_img = np.uint8(cv2.resize(np.float32(cropped_image), resized_img_shape))
        patchcore_amap_shape = patchcore_amap.shape[:2]        
        resized_img = resized_img[resized_img_shape[0]//2 - patchcore_amap_shape[0]//2:resized_img_shape[0]//2 + patchcore_amap_shape[0]//2, \
            resized_img_shape[1]//2 - patchcore_amap_shape[1]//2:resized_img_shape[1]//2 + patchcore_amap_shape[1]//2]
        resized_img_list.append(resized_img)
        
    idx = 7 # 12 for transistor, 8 for pill, 10 for tile
    
    category_patchcore_pxl_gt_np = np.array(patchcore_gt_list[category_border[idx] : category_border[idx+1]])
    category_patchcore_amap_np = np.expand_dims(np.array(patchcore_amap_list[category_border[idx] : category_border[idx+1]]), axis = 0)
    
    ensemble_patchcore_pxl_score = get_ensemble_pxl_score(category_patchcore_amap_np)
    category_patchcore_pxl_auc = roc_auc_score(category_patchcore_pxl_gt_np.ravel(), ensemble_patchcore_pxl_score.ravel())
    
    precision, recall, thresholds = precision_recall_curve(category_patchcore_pxl_gt_np.ravel(), ensemble_patchcore_pxl_score.ravel())
    f1_scores = 2*recall*precision/(recall+precision + 1e-6) 
    best_patchcore_threshold = thresholds[np.argmax(f1_scores)]
    print(f"For patchcore, Category : {category_list[idx]}")
    print('Best patchcore threshold: ', best_patchcore_threshold)
    print('Best F1-Score: ', np.max(f1_scores))
    print('Pixel auroc: ', category_patchcore_pxl_auc)
    patchcore_pxl_score_normal = ensemble_patchcore_pxl_score.ravel()[category_patchcore_pxl_gt_np.ravel() == False]
    patchcore_pxl_score_anomal = ensemble_patchcore_pxl_score.ravel()[category_patchcore_pxl_gt_np.ravel() == True]
    
    print(f"false_normal : {np.sum(patchcore_pxl_score_normal > best_patchcore_threshold)}")
    print(f"count_normal : {len(patchcore_pxl_score_normal)}")
    print(f"true_normal : {np.sum(patchcore_pxl_score_anomal < best_patchcore_threshold)}")
    print(f"count_anomal : {len(patchcore_pxl_score_anomal)}")
    
    
    category_pxl_gt_np = np.array(gt_list[category_border[idx] : category_border[idx+1]])
    category_img_gt_np = np.array([pxl_gt.sum() > 0 for pxl_gt in category_pxl_gt_np])
    category_amap_np = np.expand_dims(np.array(amap_list[category_border[idx] : category_border[idx+1]]), axis = 0)
    
    ensemble_pxl_score = get_ensemble_pxl_score(category_amap_np)    
    category_pxl_auc = roc_auc_score(category_pxl_gt_np.ravel(), ensemble_pxl_score.ravel())
    
    precision, recall, thresholds = precision_recall_curve(category_pxl_gt_np.ravel(), ensemble_pxl_score.ravel())
    f1_scores = 2*recall*precision/(recall+precision + 1e-6)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"For proposed, Category : {category_list[idx]}")
    print('Best threshold: ', best_threshold)
    print('Best F1-Score: ', np.max(f1_scores))
    print('Pixel auroc: ', category_pxl_auc)
    
    pxl_score_normal = ensemble_pxl_score.ravel()[category_pxl_gt_np.ravel() == False]
    pxl_score_anomal = ensemble_pxl_score.ravel()[category_pxl_gt_np.ravel() == True]
    
    print(f"false_normal : {np.sum(pxl_score_normal > best_threshold)}")
    print(f"count_normal : {len(pxl_score_normal)}")
    print(f"true_normal : {np.sum(pxl_score_anomal < best_threshold)}")
    print(f"count_anomal : {len(pxl_score_anomal)}")
    
    n_bins = 500
    fig = plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 25})
    ax1 = fig.add_subplot(1,2,1)
    #ax1.set_title('Patchcore', y = 1.02)
    ax1.hist((patchcore_pxl_score_anomal, patchcore_pxl_score_normal[:len(patchcore_pxl_score_normal) // 1]), bins = n_bins, density=False,  color=[(1,0,0,0.8), (0,1,0,0.8)]  , label=["anomal", "normal"], histtype='stepfilled')
    ax1.axvline(x = best_patchcore_threshold, color = 'b', linestyle='--')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    
    ax2 = fig.add_subplot(1,2,2)
    #ax2.set_title('Our proposed', y = 1.02)
    ax2.hist((pxl_score_anomal, pxl_score_normal[:len(pxl_score_normal) // 1]), bins = n_bins, density=False, color=[(1,0,0,0.8), (0,1,0,0.8)] , label=["anomal", "normal"], histtype='stepfilled')
    ax2.axvline(x = best_threshold, color = 'b', linestyle='--')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    #fig.supxlabel('Anomaly Score', y = 0.01) # default 0.01
    #fig.supylabel('Histogram', x = 0.03) # default 0.02
    plt.savefig(os.path.join(args.project_root_path, "density.png"))
    plt.show()
    

    
    
    
    