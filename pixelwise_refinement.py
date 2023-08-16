import argparse
import torch
import os
import numpy as np

from refinement import feature_generator
from refinement import evaluation
from refinement.model import create_model
from refinement.data import listData, getData
from refinement.utils.set_seed import set_seed
from refinement.utils.make_model_path import make_model_path
from refinement.utils.averagemeter import AverageMeter
from refinement.utils.print_console import print_losses

def get_args():
    parser = argparse.ArgumentParser(description='PIXELWISE_REFINEMENT')
    parser.add_argument('--category', default='bottle') # for BTAD, category is ["01, "02", 03"]
    parser.add_argument('--input_data_path', default=os.path.join('result','ensemble_ravel'), help="convert result format of main algorithm")
    parser.add_argument('--output_data_path', default=os.path.join('result','refined_ravel'), help="output directory for refined results")
    parser.add_argument('--size_patch_gt', default=(240, 240))
    parser.add_argument('--offset_patch_gt', default=0)
    parser.add_argument('--size_patch_mid', default=(240, 240))
    parser.add_argument('--offset_patch_mid', default=0)
    parser.add_argument('--size_pred', default=(256, 256))
    parser.add_argument('--offset_pred', default=8)
    parser.add_argument('--weight_refine', default=0.1)
    parser.add_argument('--size_save', default=(480, 480))
    parser.add_argument('--evaluation', default=True)
    parser.add_argument('--get_aupro', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_data_path, exist_ok=True)
    list_patch_mid, list_patch_gt, list_image_gt = listData(args.input_data_path)

    """ ###################
    #### Parameter set ####
    ################### """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    MODE = 'EVAL'
    backbone = 'densenet161_early_fusion'
    decoder_scale = 768
    epochs = 1
    lr = 0.0001
    batch_size = 8
    input_size = [320,320]
    seed = 0
    model_dir = 'data-202303050338'
    save_period = 55000
    crop_method = 'CENTER_CROP'
    loss_type = 'L2'
    conversion_coeff = 0
    use_pretrained = False
    path_pretrained = os.path.join('refinement','runs','model_refinement.pth')


    """ ###################
    #### Main function #### 
    ################### """
    set_seed(seed)

    # loading training/testing data
    train_loader, num_train_data \
        = getData(list_patch_mid, list_patch_gt, list_image_gt, batch_size=batch_size, MODE=MODE, input_size=input_size)

    model_name = backbone
    model = create_model(model_name, decoder_scale)
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # iter/epoch
    iter_per_epoch = len(train_loader)

    # weight arrangement argument
    previous_total_loss = []
    previous_loss = []

    if MODE == 'TRAIN':
        previous_total_loss.append(0)
        previous_loss.append(0)

    # 0 epoch model
    if MODE == 'TRAIN':
        # Model path
        model_path = make_model_path(model_name, decoder_scale, batch_size)
        try:
            if use_pretrained == True:
                model_name = path_pretrained
            else:
                # try to load iter00000
                model_name = "iter00000.pth"
            model.load_state_dict(torch.load(model_name))
            print('LOAD MODEL ', model_name)
        except:
            # save model
            print('THERE IS NO MODEL TO LOAD')
            model_name = model_path + "/" + 'iter' + str(0).zfill(7) + ".pth"
            print('SAVE MODEL:' + model_path)
            torch.save(model.state_dict(), model_name)
        current_iter = 0

    # Start training...
    for epoch in range(epochs):
        if MODE == 'EVAL':
            current_iter = (epoch+1) * save_period
            model_path = os.path.join('refinement','runs', model_dir)
            model_name = os.path.join('refinement','runs', model_dir, 'iter' + str(current_iter).zfill(7) + '.pth')
            model.load_state_dict(torch.load(model_name))
            print('EVAL  ' + model_name)

        if MODE == 'TRAIN':
            model.train()
        elif MODE == 'EVAL':
            model.eval()

        print('---------------------------------------------------------')
        print('-------------- TRAINING OF EPOCH ' + str(0 + epoch + 1).zfill(2) + 'START ----------------')
        batch_time = AverageMeter()
        
        current_iter, series_loss, series_l_patch, series_l_patch_dx, series_l_patch_dy \
            = feature_generator.run(
            args,
            model=model,
            model_path=model_path,
            output_path=args.output_data_path,
            optimizer=optimizer,
            loss_type=loss_type,
            conversion_coeff=conversion_coeff,
            data_loader=train_loader,
            MODE=MODE,
            crop_method=crop_method,
            epoch=epoch,
            batch_time=batch_time,
            iter_per_epoch=iter_per_epoch,
            save_period=save_period,
            current_iter=current_iter)

        if args.evaluation == True:
            evaluation.run(args)

        mean_loss = np.asarray(series_loss).mean()
        mean_l_patch = np.asarray(series_l_patch).mean()
        mean_l_patch_dx = np.asarray(series_l_patch_dx).mean()
        mean_l_patch_dy = np.asarray(series_l_patch_dy).mean()

        print_losses(epoch, mean_loss, mean_l_patch, mean_l_patch_dx, mean_l_patch_dy)