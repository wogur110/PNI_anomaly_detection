import pytorch_lightning as pl
import argparse
import os

from utils.data.load_data import Train_Dataloader, Test_Dataloader, Distribution_Train_Dataloader, Coor_Distribution_Train_Dataloader
from utils.learning.train_part import Coreset, Distribution, AC_Model, Coor_Distribution
from pytorch_lightning.loggers import TensorBoardLogger

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--train_coreset', default=False, action='store_true', help="Whether to train embedding coreset and distribution coreset")
    parser.add_argument('--train_nb_dist', default=False, action='store_true', help="Whether to train normal feature distribution from neighborhood information")
    parser.add_argument('--train_coor_dist', default=False, action='store_true', help="Whether to train normal feature distribution from position information")
    parser.add_argument('--dataset_path', default='../dataset/MVTecAD', help="root directory of dataset")
    parser.add_argument('--dataset_category', choices=['MVTec', 'BTAD'], default='MVTec', help="select type of dataset") # MVTec
    parser.add_argument('--category', default='hazelnut') # for BTAD, category is ["01, "02", 03"]
    parser.add_argument('--project_root_path', default=r'./result', help="default directory of result")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--num_workers', default=0) # 0
    
    # patch_core
    parser.add_argument('--backbone', '-b', choices=['WR101', 'WR50', 'R50', 'R34', 'R18', 'R101', 'R152', 'RNX101', 'DN201'], default='WR101') # pretrained model with ImageNet
    parser.add_argument('--layer_index', '-le', nargs='+', default=['layer2', 'layer3']) # intermediate layers to make local features
    parser.add_argument('--pretrain_embed_dimension', type=int, default=1024) # Dimensionality of features extracted from backbone layers
    parser.add_argument('--target_embed_dimension', type=int, default=1024) # final aggregated PatchCore Dimensionality
    parser.add_argument('--anomaly_nn', type=int, default=3) # Num. nearest neighbours to use for anomaly detection
    parser.add_argument('--patchsize', type=int, default=5) # neighbourhoodsize for local aggregation
    
    # sampler
    parser.add_argument('--subsampling_percentage', '-p', type=float, default=0.01) # subsampling percentage to make embedding coreset
    
    # dataset
    parser.add_argument('--resize', type=int, default=512, help='resolution of resize')
    parser.add_argument('--imagesize', type=int, default=480, help='resolution of centercrop')
    
    # coreset_distribution
    parser.add_argument('--dist_coreset_size', type=int, default=2048) # size of distribution coreset
    parser.add_argument('--dist_padding', type=int, default=4) # patch size of neighborhood features // 2
    parser.add_argument('--num_layers', type=int, default=10) # num of layers of MLP network
    parser.add_argument('--num_epochs', type=int, default=15) # train epochs of MLP network
    parser.add_argument('--learning_rate', type=float, default=1e-3) # learning rate of MLP network
    parser.add_argument('--step_size', type=int, default=5) # step size of StepLR scheduler
    parser.add_argument('--dist_batchsize', type=int, default=2048) # batch size to train MLP network
    parser.add_argument('--softmax_temperature_alpha', type=float, default=2.0)  # temperature for temperature scaling
    parser.add_argument('--prob_gamma', type=float, default=0.99) # hyperparameter for calculating probabilty of normal feature
    parser.add_argument('--softmax_nb_gamma', type=float, default=0.5) # threshold hyperparameter w.r.t. neighborhood information 
    parser.add_argument('--softmax_coor_gamma', type=float, default=0.5) # threshold hyperparameter w.r.t. position information 
    parser.add_argument('--blursigma', type=float, default=8.0) # gaussian blur sigma after resizing anomaly map
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(args.seed)
    default_root_dir = os.path.join(args.project_root_path, args.category, args.backbone) # ./MVTec/hazelnut/WR101
    
    if args.dataset_category == "MVTec" :
        args.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(args.layer_index)}', args.category, args.backbone)
    elif args.dataset_category == "BTAD" :
        args.embedding_dir_path = os.path.join('./', 'BTAD_embedding', f'embeddings_{"+".join(args.layer_index)}', args.category, args.backbone)
    os.makedirs(args.embedding_dir_path, exist_ok=True)
    
    if args.backbone == 'DN201' :        
        args.layer_index = [layer_index.replace('layer', 'features.denseblock') for layer_index in args.layer_index]

    # generate train dataloader and test dataloader from dataset
    train_dataloader, test_dataloader = Train_Dataloader(args), Test_Dataloader(args)

    # generate embedding coreset and distribution coreset and save it to faiss
    if args.train_coreset :
        print("Start generating embeeding coreset and distribution coreset")
        coreset_generator_trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(default_root_dir, 'coreset'), max_epochs=1, gpus=1, enable_checkpointing=False)
        coreset_generator = Coreset(args)
        coreset_generator_trainer.fit(coreset_generator, train_dataloaders=train_dataloader)
        print("End generating embeeding coreset and distribution coreset")

    # generate Distribution train dataloader for training coreset distribution
    distribution_train_dataloader, distribution_val_dataloader, dist_input_size, dist_output_size = Distribution_Train_Dataloader(args, train_dataloader)
    
    # train normal feature distribution from neighbrohood information
    if args.train_nb_dist:
        print("Start training normal feature distribution from neighbrohood information")
        tb_logger = TensorBoardLogger(save_dir=default_root_dir, name="distribution")
        distribution_trainer = pl.Trainer.from_argparse_args(args, max_epochs=args.num_epochs, gpus=1, logger=tb_logger, log_every_n_steps=50, enable_checkpointing=False) 
        distribution_model = Distribution(args, dist_input_size, dist_output_size)
        distribution_trainer.fit(distribution_model, train_dataloaders=distribution_train_dataloader, val_dataloaders=distribution_val_dataloader)
        print("End training normal feature distribution from neighbrohood information")
    
    # train normal feature distribution from position information
    if args.train_coor_dist:
        print("Start training normal feature distribution from position information")
        coor_distribution_train_dataloader, coor_dist_input_size, coor_dist_output_size = Coor_Distribution_Train_Dataloader(args, train_dataloader, with_edge = False)
        coor_distribution_trainer = Coor_Distribution(args, coor_dist_input_size, coor_dist_output_size, with_edge=False)
        coor_distribution_trainer.fit(coor_distribution_train_dataloader)
        coor_distribution_train_dataloader, coor_dist_input_size, coor_dist_output_size = Coor_Distribution_Train_Dataloader(args, train_dataloader, with_edge = True)
        coor_distribution_trainer = Coor_Distribution(args, coor_dist_input_size, coor_dist_output_size, with_edge=True)
        coor_distribution_trainer.fit(coor_distribution_train_dataloader)
        print("End training normal feature distribution from position information")

    # inference, evaluate anomaly score from test_dataloader
    print("Start evaluating anomaly score")
    anomaly_calculator = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(default_root_dir, 'anomaly'), max_epochs=1, gpus=1, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    ac_model = AC_Model(args, dist_input_size, dist_output_size)
    anomaly_calculator.test(ac_model, dataloaders=test_dataloader)
    print("End evaluating anomaly score")