# train our proposed model and evaluates anomaly map for each category
# You can change dataset with "--dataset_category" argument, and category with "--category" path. 
# Note that dataset should be in directory of "--dataset_path" argument. 
# If you want other pretrained network rather than WideResNet101-2, change "--backbone" argument.

python train_coreset_distribution.py --category bottle --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category cable --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category capsule --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category carpet --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category grid --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category hazelnut --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category leather --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category metal_nut --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category pill --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category screw --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category tile --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category toothbrush --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category transistor --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category wood --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category zipper --seed 23 --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD

# make ensemble score for each category and save the result in "./result/ensemble_result" repository.
#"--backbone_list" argument is list of pretrained networks which are to ensemble. You can change category with "--category" path. 
python analysis_code/calc_ensemble_score.py --category bottle --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category cable --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category capsule --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category carpet --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category grid --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category hazelnut --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category leather --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category metal_nut --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category pill --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category screw --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category tile --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category toothbrush --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category transistor --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category wood --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result
python analysis_code/calc_ensemble_score.py --category zipper --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/WR101_result

# convert result format and save it into "./result/ensemble_ravel" repository.
# Add argument "--is_BTAD" if dataset is BTAD, and "--is_MVtec_small" if dataset is small version of MVTec which we provided.
# Default dataste is MVTec AD benchmark.
python analysis_code/convert_result_format.py --before_result_root_dir ./result/WR101_result --after_result_root_dir ./result/WR101_ravel

# analysis anomaly map from "./result/ensemble_ravel" repository.
# Add argument "--visualize" to visualize anomaly map on "./result/ensemble_ravel/viz" repository.
# If you want to find misclassified images with trained model, add argument "--calc_misclassified_sample" and indices of false positive samples and false negative samples will be presented on "./result/ensemble_ravel/misclassified_sample_list.csv"
# In addition, add "--calc_pro" argument to additionally calculate AUPRO score. The result will presented on "./result/ensemble_ravel/score_result.csv".
python analysis_code/analysis_amap.py --project_root_path ./result/WR101_ravel --visualize