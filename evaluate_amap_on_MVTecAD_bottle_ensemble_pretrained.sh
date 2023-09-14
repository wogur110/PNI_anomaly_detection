# use pretrained proposed model and evaluates anomaly map for each category
# You can change dataset with "--dataset_category" argument, and category with "--category" path. 
# Note that dataset should be in directory of "--dataset_path" argument. 
# If you want other pretrained network rather than WideResNet101-2, change "--backbone" argument.

python train_coreset_distribution.py --category bottle --seed 23 --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category bottle -b RNX101 --dataset_path ../dataset/MVTecAD
python train_coreset_distribution.py --category bottle -b DN201 --dataset_path ../dataset/MVTecAD

# make ensemble score for each category and save the result in "./result/ensemble_result" repository.
#"--backbone_list" argument is list of pretrained networks which are to ensemble. You can change category with "--category" path. 
python analysis_code/calc_ensemble_score.py --category bottle --backbone_list WR101 RNX101 DN201 --project_root_path ./result --ensemble_root_path ./result/ensemble_result