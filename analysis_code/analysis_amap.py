import pytorch_lightning as pl
import argparse
import os
from pathlib import Path
import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import cv2
from PIL import Image
from skimage import measure
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--project_root_path', default=r'../result/ensemble_ravel')
    parser.add_argument('--calc_pro', default=False, action='store_true', help="Whether to calcutate pro score, which is time consuming")
    parser.add_argument('--compare_with_patchcore', default=False, action='store_true', help="Whether to compare score with patchcore")
    parser.add_argument('--visualize', default=False, action='store_true', help="Whether to visualize anomaly map image")
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--calc_misclassified_sample', default=False, action='store_true', help="Whether to calculate misclassified sample")
    parser.add_argument('--is_BTAD', default = False, action="store_true", help="Whether to use BTAD dataset")
    parser.add_argument('--is_MVTec_small', default = False, action="store_true", help="Whether to use MVTec_small dataset")

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
    
def min_max_norm_v2(image, max_value):
    a_min, a_max = image.min(), image.max()

    restricted = (image - a_min)/(max_value - a_min)
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

def image_with_ground_truth(img, gt, kernel_size = (7, 7)):
    kernel = np.ones(kernel_size, np.uint8)
    gt = np.float32(gt)
    erode_gt = cv2.erode(gt, kernel)
    dilate_gt = cv2.dilate(gt, kernel)
    edge_gt = dilate_gt - erode_gt
    
    red_edge_gt = np.zeros(shape=(edge_gt.shape + (3,)))
    red_edge_gt[:, :, 1] = 128 * edge_gt
    red_edge_gt[:, :, 2] = 255 * edge_gt
    
    img_part_mask = np.zeros_like(red_edge_gt)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_gt < 0.1)
    
    out = red_edge_gt + img * img_part_mask
    return np.uint8(out)

def image_with_predicted_mask(img, amap, threshold, kernel_size = (7, 7)) :
    predicted_mask = np.float32(amap > threshold)
    kernel = np.ones(kernel_size, np.uint8)
    erode_mask = cv2.erode(predicted_mask, kernel)
    dilate_mask = cv2.dilate(predicted_mask, kernel)
    edge_mask = dilate_mask - erode_mask
    
    red_edge_mask = np.zeros(shape=(edge_mask.shape + (3,)))
    red_edge_mask[:, :, 1] = 128 * edge_mask
    red_edge_mask[:, :, 2] = 255 * edge_mask
    
    img_part_mask = np.zeros_like(red_edge_mask)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_mask < 0.1)
    
    out = red_edge_mask + img * img_part_mask
    return np.uint8(out)
    
    
def heatmap_on_image_with_prediced_mask(heatmap_on_img, amap, threshold, kernel_size = (7, 7)) :
    predicted_mask = np.float32(amap > threshold)
    kernel = np.ones(kernel_size, np.uint8)
    erode_mask = cv2.erode(predicted_mask, kernel)
    dilate_mask = cv2.dilate(predicted_mask, kernel)
    edge_mask = dilate_mask - erode_mask
    
    red_edge_mask = np.zeros(shape=(edge_mask.shape + (3,)))
    red_edge_mask[:, :, 1] = 128 * edge_mask
    red_edge_mask[:, :, 2] = 255 * edge_mask
    
    img_part_mask = np.zeros_like(red_edge_mask)
    img_part_mask[:, :, 0] = img_part_mask[:, :, 1] = img_part_mask[:, :, 2] = (edge_mask < 0.1)
    
    out = red_edge_mask + heatmap_on_img * img_part_mask
    return np.uint8(out)

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def cal_pro_metric_new(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
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
    for step in tqdm(range(max_steps)):
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
        
if __name__ == '__main__':
    args = get_args()
    print(f"Start calculating anomaly score!")
    ensemble_result_dir = Path(os.path.join(args.project_root_path, "viz"))
    ensemble_result_dir.mkdir(parents=True, exist_ok=True)
    
    category_border = [0, 83, 233, 365, 482, 560, 670, 794, 909, 1076, 1236, 1353, 1395, 1495, 1574, 1725]
    category_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    if args.is_MVTec_small :
        category_border = [0, 25]
        category_list = ['bottle']
    if args.is_BTAD :
        category_border = [0, 70, 300, 741]
        category_list = ['01', '02', '03']
    
    gt_list = []
    amap_list = [] 
    img_list = []
    resized_img_list = []
    gt_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "*_gt.png")))
    amap_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "*_pred.png")))
    img_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "image*.png")))
    if args.compare_with_patchcore :
        patchcore_gt_list = []
        patchcore_amap_list = [] 
        patchcore_amap_path_list = sorted(glob.glob(os.path.join(args.project_root_path, "patchcore", "*.png")))
        patchcore_threshold_list = []
        patchcore_max_list = []
        
    ensemble_img_score_list = []
    pxl_auc_score_list = []
    img_auc_score_list = []
    pxl_pro_score_list = []
    threshold_list = []
    img_threshold_list = []
    max_list = []
    
    
    for idx, gt_path in enumerate(gt_path_list) :
        amap_path = amap_path_list[idx]
        img_path = img_path_list[idx]
        if args.compare_with_patchcore :
            patchcore_amap_path = patchcore_amap_path_list[idx]
        gt = (np.array(Image.open(gt_path)) / 65535).astype(int) 
        
        gt_list.append((np.array(Image.open(gt_path)) / 65535).astype(int))
        amap = np.array(Image.open(amap_path)) / 65535
        amap_list.append(amap)
        if args.compare_with_patchcore :
            patchcore_amap = np.array(Image.open(patchcore_amap_path)) / 65535
            patchcore_amap_list.append(patchcore_amap)
            patchcore_gt_list.append(gt)
        
            # resized_gt_shape = (240, 240)
            # resized_gt = np.uint8(cv2.resize(np.float32(gt), resized_gt_shape, interpolation=cv2.INTER_NEAREST))
            # patchcore_amap_shape = patchcore_amap.shape[:2]        
            # cropped_gt = resized_gt[resized_gt_shape[0]//2 - patchcore_amap_shape[0]//2:resized_gt_shape[0]//2 + patchcore_amap_shape[0]//2, \
            #     resized_gt_shape[1]//2 - patchcore_amap_shape[1]//2:resized_gt_shape[1]//2 + patchcore_amap_shape[1]//2]
            # patchcore_gt_list.append(cropped_gt)
            
        origin_image = np.array(Image.open(img_path))
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
        if len(origin_image.shape) == 2:
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_GRAY2BGR)
        
        amap_shape = amap.shape[:2]
        origin_shape = origin_image.shape[:2]
        cropped_image = origin_image[origin_shape[0]//2 - amap_shape[0]//2:origin_shape[0]//2 + amap_shape[0]//2, \
            origin_shape[1]//2 - amap_shape[1]//2:origin_shape[1]//2 + amap_shape[1]//2]
        
        img_list.append(cropped_image)
        
        if args.compare_with_patchcore :
            # resized_img_shape = (240, 240)
            # resized_img = np.uint8(cv2.resize(np.float32(cropped_image), resized_img_shape))
            # patchcore_amap_shape = patchcore_amap.shape[:2]        
            # resized_img = resized_img[resized_img_shape[0]//2 - patchcore_amap_shape[0]//2:resized_img_shape[0]//2 + patchcore_amap_shape[0]//2, \
            #     resized_img_shape[1]//2 - patchcore_amap_shape[1]//2:resized_img_shape[1]//2 + patchcore_amap_shape[1]//2]
            # resized_img_list.append(resized_img)
            resized_img_list.append(cropped_image)
    
    # for patchcore
    if args.compare_with_patchcore :
        for idx in range(len(category_list)) :
            category_patchcore_pxl_gt_np = np.array(patchcore_gt_list[category_border[idx] : category_border[idx+1]])
            category_patchcore_amap_np = np.expand_dims(np.array(patchcore_amap_list[category_border[idx] : category_border[idx+1]]), axis = 0)
            
            ensemble_patchcore_pxl_score = get_ensemble_pxl_score(category_patchcore_amap_np)
            
            precision, recall, thresholds = precision_recall_curve(category_patchcore_pxl_gt_np.ravel(), ensemble_patchcore_pxl_score.ravel())
            f1_scores = 2*recall*precision/(recall+precision + 1e-6)
            best_patchcore_threshold = thresholds[np.argmax(f1_scores)]
            print(f"Category : {category_list[idx]}")
            print('Best patchcore threshold: ', best_patchcore_threshold)
            print('Best F1-Score: ', np.max(f1_scores))
            patchcore_threshold_list.append(best_patchcore_threshold)
            patchcore_max_list.append(np.max(ensemble_patchcore_pxl_score))

    for idx in range(len(category_list)) :
        category_pxl_gt_np = np.array(gt_list[category_border[idx] : category_border[idx+1]])
        category_img_gt_np = np.array([pxl_gt.sum() > 0 for pxl_gt in category_pxl_gt_np])
        category_amap_np = np.expand_dims(np.array(amap_list[category_border[idx] : category_border[idx+1]]), axis = 0)
        
        ensemble_pxl_score = get_ensemble_pxl_score(category_amap_np)
        ensemble_img_score = get_ensemble_img_score(category_amap_np)
        
        category_pxl_auc = roc_auc_score(category_pxl_gt_np.ravel(), ensemble_pxl_score.ravel())
        category_img_auc = roc_auc_score(category_img_gt_np, ensemble_img_score)
        if args.calc_pro :
            category_pxl_pro = cal_pro_metric_new(category_pxl_gt_np, ensemble_pxl_score, max_steps=args.max_steps)
            pxl_pro_score_list.append(category_pxl_pro)
        
        img_auc_score_list.append(category_img_auc)
        pxl_auc_score_list.append(category_pxl_auc)        
        
        precision, recall, thresholds = precision_recall_curve(category_pxl_gt_np.ravel(), ensemble_pxl_score.ravel())
        f1_scores = 2*recall*precision/(recall+precision + 1e-6)
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Category : {category_list[idx]}")
        print('Best threshold: ', best_threshold)
        print('Best F1-Score: ', np.max(f1_scores))
        threshold_list.append(best_threshold)
        
        precision, recall, thresholds = precision_recall_curve(category_img_gt_np.ravel(), ensemble_img_score.ravel())
        f1_scores = 2*recall*precision/(recall+precision + 1e-6)
        best_img_threshold = thresholds[np.argmax(f1_scores)]
        print('Best Img threshold: ', best_img_threshold)
        img_threshold_list.append(best_img_threshold)
        ensemble_img_score_list.extend(ensemble_img_score)
        
        max_list.append(np.max(ensemble_pxl_score))
        
        print("Image AUROC: ", f'{category_img_auc : .6f}')
        print("Pixel AUROC: ", f'{category_pxl_auc : .6f}')
        if args.calc_pro :
            print("Pixel AUPRO: ", f'{category_pxl_pro : .6f}')
        print("")
        
        f = open(os.path.join(args.project_root_path, f"score_result.csv"), "a")
        data = [category_list[idx], str(f'{category_img_auc : .6f}'), str(f'{category_pxl_auc : .6f}'), str(f'{best_threshold : .6f}')]
        if args.calc_pro :
            data.append(str(f'{category_pxl_pro : .6f}'))
        data = ','.join(data) + '\n'
        f.write(data)
        f.close()
        
    avg_img_auc = np.mean(img_auc_score_list)
    avg_pxl_auc = np.mean(pxl_auc_score_list)
    if args.calc_pro :
        avg_pxl_pro = np.mean(pxl_pro_score_list)
    
    f = open(os.path.join(args.project_root_path, f"score_result.csv"), "a")
    data = ["Avg", str(f'{avg_img_auc : .6f}'), str(f'{avg_pxl_auc : .6f}')]
    if args.calc_pro :
        data.append(str(f'{avg_pxl_pro : .6f}'))
    data = ','.join(data) + '\n'
    f.write(data)
    f.close()
    
    if args.calc_misclassified_sample :
        false_negative_list = []
        false_positive_list = []
        
        print(f"Start calculate misclassified samples!")
        for idx, gt in enumerate(gt_list) :
            amap = amap_list[idx]
            img = img_list[idx]
            ensemble_img_score = ensemble_img_score_list[idx]
            
            for category_idx in range(len(category_list)) :
                if category_border[category_idx] <= idx and idx < category_border[category_idx + 1]:
                    category = category_list[category_idx]
                    threshold = threshold_list[category_idx]
                    img_threshold = img_threshold_list[category_idx]
                    maximum = max_list[category_idx]
                    break
            
            if ensemble_img_score < img_threshold and gt.sum() > 0 :
                false_negative_list.append(str(idx) + ',' + str(category) + ',' + str(f'{ensemble_img_score : .6f}') + ',' + str(f'{img_threshold : .6f}'))
            if ensemble_img_score > img_threshold and gt.sum() == 0 :
                false_positive_list.append(str(idx) + ',' + str(category) + ',' + str(f'{ensemble_img_score : .6f}') + ',' + str(f'{img_threshold : .6f}'))
        
        f = open(os.path.join(args.project_root_path, f"misclassified_sample_list.csv"), "a")
        f.write("False Negative List\n")
        f.write('\n'.join(false_negative_list) + '\n')
        f.write("False Positive List\n")
        f.write('\n'.join(false_positive_list) + '\n')
        f.close()
    
    if args.visualize :
        print(f"Start Visualize Result!")
        
        for idx, gt in enumerate(gt_list) :
            amap = amap_list[idx]
            img = img_list[idx]
            if args.compare_with_patchcore :
                resized_img = resized_img_list[idx]
                patchcore_amap = patchcore_amap_list[idx]
            
            for category_idx in range(len(category_list)) :
                if category_border[category_idx] <= idx and idx < category_border[category_idx + 1]:
                    category = category_list[category_idx]
                    threshold = threshold_list[category_idx]
                    img_threshold = img_threshold_list[category_idx]
                    maximum = max_list[category_idx]
                    if args.compare_with_patchcore :
                        patchcore_threshold = patchcore_threshold_list[category_idx]
                        patchcore_maximum = patchcore_max_list[category_idx]
                    break        
            gt_path = gt_path_list[idx]
            
            fname = gt_path.split("/")[-1]
            fnum = fname.split("_")[0]
            
            Path(os.path.join(ensemble_result_dir, category)).mkdir(parents=True, exist_ok=True)
            
            img_path = os.path.join(ensemble_result_dir, category, f"{fnum}.png")
            gt_path = os.path.join(ensemble_result_dir, category, f"{fnum}_mask.png")
            amap_path = os.path.join(ensemble_result_dir, category, f"{fnum}_amap.png")
            amap_on_img_path = os.path.join(ensemble_result_dir, category, f"{fnum}_amap_on_img.png")
            img_with_gt_path = os.path.join(ensemble_result_dir, category, f"{fnum}_img_with_gt.png")
            img_with_mask_path = os.path.join(ensemble_result_dir, category, f"{fnum}_img_with_mask.png")
            heatmap_on_img_with_mask_path = os.path.join(ensemble_result_dir, category, f"{fnum}_heatmap_on_img_with_mask.png")
            if args.compare_with_patchcore :
                patchcore_amap_path = os.path.join(ensemble_result_dir, category, f"{fnum}_patchcore_amap.png")
                patchcore_amap_on_img_path = os.path.join(ensemble_result_dir, category, f"{fnum}_patchcore_amap_on_img.png")
                img_with_patchcore_mask_path = os.path.join(ensemble_result_dir, category, f"{fnum}_img_with_patchcore_mask.png")
                heatmap_on_img_with_patchcore_mask_path = os.path.join(ensemble_result_dir, category, f"{fnum}_heatmap_on_img_with_patchcore_mask.png")
            
            heatmap = cvt2heatmap(min_max_norm_v2(amap, max_value = maximum) * 255)
            #heatmap = cvt2heatmap(min_max_norm(amap, thres=threshold) * 255)
            heatmap_on_img = heatmap_on_image(heatmap, img)
            img_with_gt = image_with_ground_truth(img, gt)
            img_with_mask = image_with_predicted_mask(img, amap, threshold)
            heatmap_on_img_with_mask = heatmap_on_image_with_prediced_mask(heatmap_on_img, amap, threshold)
            # img_with_mask = image_with_predicted_mask(img, amap, img_threshold)
            # heatmap_on_img_with_mask = heatmap_on_image_with_prediced_mask(heatmap_on_img, amap, img_threshold)
            
            if args.compare_with_patchcore :
                patchcore_heatmap = cvt2heatmap(min_max_norm_v2(patchcore_amap, max_value = patchcore_maximum) * 255)
                #patchcore_heatmap = cvt2heatmap(min_max_norm(patchcore_amap, thres=patchcore_threshold) * 255)
                patchcore_heatmap_on_img = heatmap_on_image(patchcore_heatmap, resized_img)
                # img_with_patchcore_mask = image_with_predicted_mask(resized_img, patchcore_amap, patchcore_threshold, kernel_size=(4,4))
                # heatmap_on_img_with_patchcore_mask = heatmap_on_image_with_prediced_mask(patchcore_heatmap_on_img, patchcore_amap, patchcore_threshold, kernel_size=(4,4))
                img_with_patchcore_mask = image_with_predicted_mask(resized_img, patchcore_amap, patchcore_threshold)
                heatmap_on_img_with_patchcore_mask = heatmap_on_image_with_prediced_mask(patchcore_heatmap_on_img, patchcore_amap, patchcore_threshold)

            cv2.imwrite(img_path, img)
            cv2.imwrite(gt_path, gt * 255)
            cv2.imwrite(amap_path, heatmap)
            cv2.imwrite(amap_on_img_path, heatmap_on_img)
            cv2.imwrite(img_with_gt_path, img_with_gt)
            cv2.imwrite(img_with_mask_path, img_with_mask)
            cv2.imwrite(heatmap_on_img_with_mask_path, heatmap_on_img_with_mask)
            
            if args.compare_with_patchcore :
                cv2.imwrite(patchcore_amap_path, patchcore_heatmap)
                cv2.imwrite(patchcore_amap_on_img_path, patchcore_heatmap_on_img)
                cv2.imwrite(img_with_patchcore_mask_path, img_with_patchcore_mask)
                cv2.imwrite(heatmap_on_img_with_patchcore_mask_path, heatmap_on_img_with_patchcore_mask)