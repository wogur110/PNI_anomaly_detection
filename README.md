# PNI : Industrial Anomaly Detection using Position and Neighborhood Information
This is supplementary code for the paper "PNI : Industrial Anomaly Detection using Position and Neighborhood Information."
It trains the model for evaluating anomaly maps and calculating Image AUROC (I-AUROC), Pixel AUROC (P-AUROC), and Pixel AUPRO for two datasets, MVTec AD benchmark, and BTAD dataset.

The model trained on designated hyperparameter can achieve up to **99.56%** and **98.98%** AUROC scores in anomaly detection and localization and **96.05%** AUPRO score for MVTec AD benchmark, which is the state-of-the-art performance.
With an ensemble of models, performance can reach **99.63%** and **99.06%** in Image AUROC and Pixel AUROC, and **96.55%** Pixel AUPRO.
In addition, the same model can achieve up to 97.8% of Pixel AUROC for the BTAD dataset, which is the highest performance compared to previous works.

## Environment
We trained and evaluated our models in Python 3.8 and PyTorch which version torch=1.12.1 and torchvision=0.13.1.
Training is on NVIDIA TITAN RTX, A100, and NVIDIA T4 GPUs.
We used ImageNet pre-trained network from PyTorch/vision:v0.10.0.
The WideResNet101-2 network is used in our code by default, ResNext101_32x8d and DenseNet201 are used for ensemble results.
In the code, the embedding coreset and the distribution coreset are stored in *faiss* framework to calculate the distance between a test feature and the coresets efficiently.
We used *pytorch-lightning* framework to manage the training and evaluation process.

## Quick Guide
We provided a bash file for training and evaluating the anomaly map for MVTec AD benchmark and BTAD dataset.
Dataset should be placed in parent directory of code repository.
For example, default dataset directory is "../dataset/MVTecAD" and "../dataset/BTAD" for MVTec AD and BTAD, respectively.

We also provided small size of MVTecAD dataset for training and evaluating the model, named MVTecAD_small. This dataset contains only 1 category, bottle, with 50 train images and 20 test images. The dataset is stored in "../dataset/MVTecAD_small" directory.

First, go to code repository, and install all requirements of environment.
The environment name we used is "anomaly_env".

```bash
conda create -y -n anomaly_env
conda activate anomaly_env 
```
```bash
conda install -y python=3.8
pip install pytorch-lightning==1.5.9
pip install pillow==9.0.0
pip install faiss-gpu==1.7.1
pip install opencv-python==4.5.2.52
pip install scikit-learn==0.24.2
pip install scikit-image==0.19.2
pip install pymp-pypi==0.5.0
pip install numpngw==0.1.2
pip install matplotlib==3.7.2
pip install timm==0.9.5
```
The available version of torch and torchvision might be different depending on hardware settings.
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

For MVTecAD dataset, execute "evaluate_anomaly_map_on_MVTecAD.sh". For MVTecAD_small dataset we provided, execute "evaluate_anomaly_map_on_MVTecAD_small.sh". This bash file contains all processes of training the model, evaluating the anomaly map, and visualizing the anomaly map.
```bash
chmod +x evaluate_anomaly_map_on_MVTecAD_small.sh
./evaluate_anomaly_map_on_MVTecAD_small.sh
```
As a result of execution, a "result" repository will be created. The structure of the repository is as follows:
```
|-- result/
  |-- bottle/
  
  |-- ensemble_ravel/
    |-- viz/
      |-- bottle/
        00000_amap.png
        00000_amap_on_img.png
        ...
    00000_gt.png
    00000_pred.png
    ...
    image00000.png
    ...
    score_result.csv
    
  |-- ensemble_result/
    |-- bottle/
      |-- anomaly_map/
      |-- ground_truth/
      |-- test/
      
  ensemble_score_result.csv
  score_result.csv
```

You can find the AUROC score on either the output terminal or "./result/score_result.csv". If you use multiple models for the ensemble, the ensemble score can be found on either the output terminal or "./result/ensemble_score_result.csv".
Visualization of the anomaly map can be found on "./result/ensemble_ravel/viz/" repository.
You can check images, ground truth, anomaly map, and anomaly map with a mask on the repository.

## Hyperparameter setting
The default hyperparameter in "evaluate_anomaly_map_on_MVTecAD_small.sh" is the same as mentioned in the paper. 
"evaluate_anomaly_map_on_MVTecAD_small.sh" contains 4 sequential python codes.

First, **"train_coreset_distribution.py"** trains our proposed model and evaluates the anomaly map for each category. You can change the dataset with "--dataset_category" argument, and the category with "--category" path. 
Note that the dataset should be in the directory of "--dataset_path" argument. 
If you want other pre-trained networks rather than WideResNet101-2, change "--backbone" argument.

Second, **"analysis_code/calc_ensemble_score.py"** makes an ensemble score for each category and saves the result in "./result/ensemble_result" repository.
"--backbone_list" argument is a list of pre-trained networks which are to ensemble. You can change the category with "--category" path. 

Third, **"analysis_code/convert_result_format.py"** converts the result format and saves it into "./result/ensemble_ravel" repository.
Add argument "--is_BTAD" if the dataset is BTAD, and "--is_MVtec_small" if the dataset is a small version of MVTec which we provided.
The default dataset is the MVTec AD benchmark.

Finally, **"analysis_code/analysis_amap.py"** analysis anomaly map from "./result/ensemble_ravel" repository.
Add argument "--visualize" to visualize the anomaly map on "./result/ensemble_ravel/viz" repository.
If you want to find misclassified images with the trained model, add argument "--calc_misclassified_sample" and indices of false positive samples and false negative samples will be presented on "./result/ensemble_ravel/misclassified_sample_list.csv"
In addition, add "--calc_pro" argument to additionally calculate the AUPRO score. The result will be presented on "./result/ensemble_ravel/score_result.csv".
