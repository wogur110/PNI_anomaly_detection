import pytorch_lightning as pl
import argparse
import os
from pathlib import Path
import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--project_root_path', default=r'../result')
    parser.add_argument('--ensemble_root_path', default=r'../result/ensemble_result')
    parser.add_argument('--save_type', default=r'nb_coor')
    parser.add_argument('--version_num', type=int, default=1)
    
    # patch_core
    parser.add_argument('--backbone_list', '-b', nargs='+', default=['WR101', 'RNX101', 'DN201']) # pretrained model with ImageNet
    
    args = parser.parse_args()
    return args

def get_scores_from_pkl(pkl_list) :
    scores = []
    for pkl in pkl_list :
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        scores.append(data)
    return scores

def get_images_from_pkl(pkl_list) :
    scores = []
    for pkl in pkl_list :
        with open(pkl, 'rb') as f:
            image = pickle.load(f)
        if len(image.shape) == 2 :
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        scores.append(image)
    return scores

def get_ensemble_pxl_score(score_np) :
    scores_ravel = score_np.reshape(score_np.shape[0], -1)
    min_scores = scores_ravel.min(axis=-1).reshape(-1, 1, 1, 1)
    max_scores = scores_ravel.max(axis=-1).reshape(-1, 1, 1, 1)
    score_np = (score_np - min_scores) / (max_scores - min_scores)
    scores = np.mean(score_np, axis=0)
    
    return scores

def get_ensemble_img_score(score_np, anomaly_percentage = 0.01) :
    backbone_num, img_num, H, W = score_np.shape
    scores_ravel = score_np.reshape(backbone_num * img_num, -1)
    
    mean_num = (int)(H * W * anomaly_percentage)
    
    scores_ravel_sorted = -np.sort(-scores_ravel, axis = 1)
    img_score = np.mean(scores_ravel_sorted[:, :mean_num], axis = 1).reshape(backbone_num, img_num)
    img_score = np.mean(img_score, axis = 0)
    
    # max_scores = scores_ravel.max(axis=1).reshape(backbone_num, img_num)
    # img_score = np.mean(max_scores, axis = 0)
    
    return img_score

def min_max_norm(image, thres=-1):
    a_min, a_max = image.min(), image.max()
    if thres == -1 :
        return (image - a_min)/(a_max - a_min)
    else :
        restricted = np.maximum((image - thres)/(a_max - thres), 0)
        return np.power(restricted, 0.5)
    
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
        
if __name__ == '__main__':
    args = get_args()
    print(f"Start ensemble {args.category}!")
    ensemble_result_dir = Path(os.path.join(args.ensemble_root_path, args.category))
    ensemble_result_dir.mkdir(parents=True, exist_ok=True)
    
    coor_score_list = []
    nb_coor_score_list = []
    nb_score_list = []
    patchcore_score_list = []
    gt_list = []
    img_list = []
    fname_list = []
    
    for idx, backbone in enumerate(args.backbone_list) : 
        default_root_dir = os.path.join(args.project_root_path, args.category, backbone, 'anomaly', 'lightning_logs') # ./MVTec/hazelnut
        result_version_list = os.listdir(default_root_dir)
        result_version_list = sorted([int(version.strip('version_')) for version in result_version_list])
        latest_version = 'version_' + str(result_version_list[-1 * args.version_num])
        
        coor_pkl_list = sorted(glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*amap_coor.pkl')))
        nb_coor_pkl_list = sorted(glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*amap_nb_coor.pkl')))
        nb_pkl_list = sorted(glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*amap_nb.pkl')))
        patchcore_pkl_list = sorted(glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*amap_patchcore.pkl')))
        gt_pkl_list = sorted(glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*_gt.pkl')))
        img_pkl_list = sorted(glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*_0???.pkl')) + glob.glob(os.path.join(default_root_dir, latest_version, "sample", '*_0??.pkl')))
        
        coor_score_list.append(get_scores_from_pkl(coor_pkl_list))
        nb_coor_score_list.append(get_scores_from_pkl(nb_coor_pkl_list))
        nb_score_list.append(get_scores_from_pkl(nb_pkl_list))
        patchcore_score_list.append(get_scores_from_pkl(patchcore_pkl_list))

        if idx == 0 :
            gt_list = get_scores_from_pkl(gt_pkl_list)
            img_list = get_scores_from_pkl(img_pkl_list)
            fname_list = [img_pkl.strip(".pkl") for img_pkl in img_pkl_list]
            
    coor_score_np = np.array(coor_score_list)   
    nb_coor_score_np = np.array(nb_coor_score_list)
    nb_score_np = np.array(nb_score_list)
    patchcore_score_np = np.array(patchcore_score_list)
    gt_np = np.array(gt_list)
    img_np = np.array(img_list)
    img_gt_np = np.array([pxl_gt.sum() > 0 for pxl_gt in gt_np])
    
    coor_ensemble_pxl_score = get_ensemble_pxl_score(coor_score_np)
    nb_coor_ensemble_pxl_score = get_ensemble_pxl_score(nb_coor_score_np)
    nb_ensemble_pxl_score = get_ensemble_pxl_score(nb_score_np)
    patchcore_ensemble_pxl_score = get_ensemble_pxl_score(patchcore_score_np)
    
    coor_ensemble_img_score = get_ensemble_img_score(coor_score_np)
    nb_coor_ensemble_img_score = get_ensemble_img_score(nb_coor_score_np)
    nb_ensemble_img_score = get_ensemble_img_score(nb_score_np)
    patchcore_ensemble_img_score = get_ensemble_img_score(patchcore_score_np)
    
    # Total pixel-level auc-roc score
    pixel_auc_nb = roc_auc_score(gt_np.ravel(), nb_ensemble_pxl_score.ravel())
    # Total pixel-level auc-roc score for only using likelihood
    pixel_auc_coor = roc_auc_score(gt_np.ravel(), coor_ensemble_pxl_score.ravel())
    # Total pixel-level auc-roc score for patchcore version
    pixel_auc_patchcore = roc_auc_score(gt_np.ravel(), patchcore_ensemble_pxl_score.ravel())
    # Total pixel-level auc-roc score
    pixel_auc_nb_coor = roc_auc_score(gt_np.ravel(), nb_coor_ensemble_pxl_score.ravel())
    
    # Total image-level auc-roc score
    img_auc_nb = roc_auc_score(img_gt_np, nb_ensemble_img_score)
    # Total image-level auc-roc score for only using likelihood
    img_auc_coor = roc_auc_score(img_gt_np, coor_ensemble_img_score)
    # Total image-level auc-roc score for patchcore version
    img_auc_patchcore = roc_auc_score(img_gt_np, patchcore_ensemble_img_score)
    # Total image-level auc-roc score
    img_auc_nb_coor = roc_auc_score(img_gt_np, nb_coor_ensemble_img_score)
    
    f = open(os.path.join(args.project_root_path, "ensemble_score_result.csv"), "a")
    data = [args.category, \
            str(f'{pixel_auc_nb : .6f}'), str(f'{pixel_auc_coor : .6f}'), str(f'{pixel_auc_patchcore : .6f}'), str(f'{pixel_auc_nb_coor : .6f}'),\
            str(f'{img_auc_nb : .6f}'), str(f'{img_auc_coor : .6f}'), str(f'{img_auc_patchcore : .6f}'), str(f'{img_auc_nb_coor : .6f}')]
    data = ','.join(data) + '\n'
    f.write(data)
    f.close()
    
    print(f"End ensemble {args.category}!")
    
    print(f"Start save ensemble {args.category}!")
    
    for idx, fname in enumerate(fname_list) :
        fname = fname.split("/")[-1]
        fnum = fname.split("_")[-1]
        ftype = fname.strip(f"_{fnum}")
        
        Path(os.path.join(ensemble_result_dir, "test", ftype)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(ensemble_result_dir, "ground_truth", ftype)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(ensemble_result_dir, "anomaly_map", ftype)).mkdir(parents=True, exist_ok=True)
        
        gt = gt_np[idx]
        img = img_np[idx]
        if args.save_type == "nb_coor" :
            amap = nb_coor_ensemble_pxl_score[idx]
        elif args.save_type == "nb" :
            amap = nb_ensemble_pxl_score[idx]
        elif args.save_type == "coor" :
            amap = coor_ensemble_pxl_score[idx]
        elif args.save_type == "patchcore" :
            amap = patchcore_ensemble_pxl_score[idx]
        
        img_path = os.path.join(ensemble_result_dir, "test", ftype, f"{fnum}.png")
        gt_path = os.path.join(ensemble_result_dir, "ground_truth", ftype, f"{fnum}_mask.png")
        amap_pkl_path = os.path.join(ensemble_result_dir, "anomaly_map", ftype, f"{fnum}.pkl")
        amap_png_path = os.path.join(ensemble_result_dir, "anomaly_map", ftype, f"{fnum}.png")

        cv2.imwrite(img_path, img)
        cv2.imwrite(gt_path, gt * 255)
        cv2.imwrite(amap_png_path, cvt2heatmap(min_max_norm(amap) * 255))
        with open(amap_pkl_path, "wb") as fp :
            pickle.dump(amap, fp)