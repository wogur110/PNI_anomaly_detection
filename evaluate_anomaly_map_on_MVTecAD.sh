# train our proposed model and evaluates anomaly map for each category
# You can change dataset with "--dataset_category" argument, and category with "--category" path. 
# Note that dataset should be in directory of "--dataset_path" argument. 
# If you want other pretrained network rather than WideResNet101-2, change "--backbone" argument.

python train_coreset_distribution.py --category bottle --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category cable --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category capsule --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category carpet --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category grid --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category hazelnut --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category leather --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category metal_nut --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category pill --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category screw --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category tile --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category toothbrush --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category transistor --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category wood --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category zipper --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTecAD

# make ensemble score for each category and save the result in "./result/ensemble_result" repository.
#"--backbone_list" argument is list of pretrained networks which are to ensemble. You can change category with "--category" path. 
python analysis_code/calc_ensemble_score.py --category bottle --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category cable --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category capsule --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category carpet --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category grid --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category hazelnut --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category leather --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category metal_nut --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category pill --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category screw --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category tile --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category toothbrush --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category transistor --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category wood --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result
python analysis_code/calc_ensemble_score.py --category zipper --backbone_list WR101 --project_root_path ./result --ensemble_root_path ./result/ensemble_result

# convert result format and save it into "./result/ensemble_ravel" repository.
# Add argument "--is_BTAD" if dataset is BTAD, and "--is_MVtec_small" if dataset is small version of MVTec which we provided.
# Default dataste is MVTec AD benchmark.
python analysis_code/convert_result_format.py --before_result_root_dir ./result/ensemble_result --after_result_root_dir ./result/ensemble_ravel

# analysis anomaly map from "./result/ensemble_ravel" repository.
# Add argument "--visualize" to visualize anomaly map on "./result/ensemble_ravel/viz" repository.
# If you want to find misclassified images with trained model, add argument "--calc_misclassified_sample" and indices of false positive samples and false negative samples will be presented on "./result/ensemble_ravel/misclassified_sample_list.csv"
# In addition, add "--calc_pro" argument to additionally calculate AUPRO score. The result will presented on "./result/ensemble_ravel/score_result.csv".
python analysis_code/analysis_amap.py --project_root_path ./result/ensemble_ravel --visualize