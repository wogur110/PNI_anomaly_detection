import argparse
import os
import glob
import pickle
from PIL import Image
import numpy as np
import numpngw
from pathlib import Path
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--before_result_root_dir', default=r'../result/ensemble_result')
    parser.add_argument('--after_result_root_dir', default=r'../result/ensemble_ravel')
    parser.add_argument('--is_BTAD', default = False, action="store_true", help="Whether to use BTAD dataset")
    parser.add_argument('--is_MVTec_small', default = False, action="store_true", help="Whether to use MVTec_small dataset")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    after_result_root_dir = Path(args.after_result_root_dir)
    after_result_root_dir.mkdir(parents=True, exist_ok=True)

    category_border = [0, 83, 233, 365, 482, 560, 670, 794, 909, 1076, 1236, 1353, 1395, 1495, 1574, 1725]
    category_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    if args.is_MVTec_small :
        category_border = [0, 25]
        category_list = ['bottle']
    if args.is_BTAD :
        category_border = [0, 70, 300, 741]
        category_list = ['01', '02', '03']

    total_amap_pkl_path_list = []
    total_gt_path_list = []
    total_img_path_list = []
    
    for category in category_list :
        category_dir = os.path.join(args.before_result_root_dir, category)
        amap_dir = os.path.join(category_dir, "anomaly_map")
        gt_dir = os.path.join(category_dir, "ground_truth")
        img_dir = os.path.join(category_dir, "test")
        type_list = sorted(os.listdir(amap_dir))
        
        for type in type_list :
            amap_type_dir = os.path.join(amap_dir, type)
            gt_type_dir = os.path.join(gt_dir, type)
            img_type_dir = os.path.join(img_dir, type)
            amap_pkl_path_list = sorted(glob.glob(os.path.join(amap_type_dir, "*.pkl")))
            gt_path_list = sorted(glob.glob(os.path.join(gt_type_dir, "*.png")))
            img_path_list = sorted(glob.glob(os.path.join(img_type_dir, "*.png")))
            total_amap_pkl_path_list.extend(amap_pkl_path_list)
            total_gt_path_list.extend(gt_path_list)
            total_img_path_list.extend(img_path_list)

    for img_idx in tqdm(range(len((total_amap_pkl_path_list)))) :
        amap_pkl_path = total_amap_pkl_path_list[img_idx]
        gt_path = total_gt_path_list[img_idx]
        img_path = total_img_path_list[img_idx]
        
        with open(amap_pkl_path, 'rb') as f:
            amap = pickle.load(f)
        gt = (np.array(Image.open(gt_path)) / 255).astype(int)
        img = np.array(Image.open(img_path))
        
        after_amap = (amap * 65535).astype(np.uint16)
        after_gt = (gt * 65535).astype(np.uint16)
        
        after_amap_path = os.path.join(after_result_root_dir, f'{str(img_idx).zfill(5)}_pred.png')
        after_gt_path = os.path.join(after_result_root_dir, f'{str(img_idx).zfill(5)}_gt.png')
        after_img_path = os.path.join(after_result_root_dir, f'image{str(img_idx).zfill(5)}.png')
        
        numpngw.write_png(after_amap_path, after_amap)
        numpngw.write_png(after_gt_path, after_gt)
        numpngw.write_png(after_img_path, img)
