import os
import time
import datetime
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image

from refinement.loss import compute_3losses

def run(args, model, model_path, output_path, optimizer, loss_type, conversion_coeff, data_loader, MODE, crop_method, epoch, batch_time, iter_per_epoch, save_period, current_iter):
    series_loss, series_l_patch, series_l_patch_dx, series_l_patch_dy = [], [], [], []
    temp_series_loss, temp_series_l_patch, temp_series_l_patch_dx, temp_series_l_patch_dy = [], [], [], []

    end = time.time()
    for i, sample_batched in enumerate(data_loader):
        optimizer.zero_grad()

        # Prepare sample and target
        image_in = sample_batched['image_in']
        patch_mid = sample_batched['patch_mid']
        image_gt = sample_batched['image_gt']
        patch_gt = sample_batched['patch_gt']

        image_in = torch.autograd.Variable(image_in.cuda())
        patch_mid = torch.autograd.Variable(patch_mid.cuda(non_blocking=True))
        image_gt = torch.autograd.Variable(image_gt.cuda())
        patch_gt = torch.autograd.Variable(patch_gt.cuda(non_blocking=True))

        #patch_pred = model(image_gt)
        if crop_method == 'FULL':
            patch_pred = patch_mid + model(image_in, patch_mid.repeat(1,3,1,1))
        elif crop_method == 'CENTER_CROP':
            up = nn.UpsamplingBilinear2d(size=[patch_mid.shape[2],patch_mid.shape[3]])
            patch_mid = up(patch_mid[:, :, 11:-11, 11:-11])
            patch_pred = patch_mid + model(image_in, patch_mid.repeat(1,3,1,1))

        # one-to-one conversion
        if conversion_coeff == 0: # conversion_coeff is infinite
            patch_pred_for_loss = patch_pred
            patch_gt_for_loss = patch_gt
        else:
            var_max = 1 / (1 + math.exp(-0.5 * conversion_coeff))
            var_min = 1 / (1 + math.exp(0.5 * conversion_coeff))
            magnification_factor = 1 / (var_max - var_min)

            sigmoid = torch.nn.Sigmoid()
            patch_pred_for_loss = magnification_factor * (sigmoid(conversion_coeff * (patch_pred - 0.5)) - var_min)
            patch_gt_for_loss = magnification_factor * (sigmoid(conversion_coeff * (patch_gt - 0.5)) - var_min)

        # compute iter loss & train_scores
        if loss_type == 'L1':
            l_patch, l_patch_dx, l_patch_dy = compute_3losses(patch_pred_for_loss, patch_gt_for_loss, n=1)
        elif loss_type == 'L2':
            l_patch, l_patch_dx, l_patch_dy = compute_3losses(patch_pred_for_loss, patch_gt_for_loss, n=2)
        elif loss_type == 'L1L2':
            l_patch_1, l_patch_dx_1, l_patch_dy_1 = compute_3losses(patch_pred_for_loss, patch_gt_for_loss, n=1)
            l_patch_2, l_patch_dx_2, l_patch_dy_2 = compute_3losses(patch_pred_for_loss, patch_gt_for_loss, n=2)
            l_patch, l_patch_dx, l_patch_dy = (l_patch_1+l_patch_1)/2, (l_patch_dx_1+l_patch_dx_2)/2, (l_patch_dy_1+l_patch_dy_2)/2
        loss = torch.mean(l_patch * 1/2 + l_patch_dx * 1/4 + l_patch_dy * 1/4)

        temp_series_loss.append(loss.item())
        temp_series_l_patch.append(torch.mean(l_patch).item())
        temp_series_l_patch_dx.append(torch.mean(l_patch_dx).item())
        temp_series_l_patch_dy.append(torch.mean(l_patch_dy).item())

        # Update step
        if MODE == 'TRAIN':
            loss.backward()
            optimizer.step()
            current_iter = current_iter + 1
        # save results
        if MODE == 'EVAL':
            if i == 0:
                index_save = 0
            for index_data in range(patch_pred.shape[0]):
                name_patch_gt = os.path.join(output_path, data_loader.sampler.data_source.list_patch_gt[index_save].replace('\\', '/').split('/')[-1])
                name_patch_mid = os.path.join(output_path, data_loader.sampler.data_source.list_patch_mid[index_save].replace('\\', '/').split('/')[-1])
                name_refine = name_patch_mid

                one_patch_gt = np.clip(65535 * patch_gt[index_data, 0].cpu().detach().numpy(), a_min=0, a_max=65535).astype(np.uint16)
                one_patch_gt = np.asarray(Image.fromarray(one_patch_gt).resize(args.size_patch_gt))
                if args.offset_patch_gt > 0:
                    offset = args.offset_patch_gt
                    one_patch_gt = one_patch_gt[offset:-offset, offset:-offset]

                one_patch_mid = np.clip(65535 * patch_mid[index_data, 0].cpu().detach().numpy(), a_min=0, a_max=65535).astype(np.uint16)
                one_patch_mid = np.asarray(Image.fromarray(one_patch_mid).resize(args.size_patch_mid))
                if args.offset_patch_mid > 0:
                    offset = args.offset_patch_mid
                    one_patch_mid = one_patch_mid[offset:-offset, offset:-offset]

                one_pred = np.clip(65535 * patch_pred[index_data, 0].cpu().detach().numpy(), a_min=0, a_max=65535).astype(np.uint16)
                one_pred = np.asarray(Image.fromarray(one_pred).resize(args.size_pred))
                if args.offset_pred > 0:
                    offset = args.offset_pred
                    one_pred = one_pred[offset:-offset, offset:-offset]

                ratio = args.weight_refine
                one_refine = (ratio * one_pred.astype(np.float32) + (1-ratio) * one_patch_mid.astype(np.float32)).astype(np.uint16)

                one_patch_gt = Image.fromarray(one_patch_gt).resize(args.size_save)
                one_refine = Image.fromarray(one_refine).resize(args.size_save)

                one_patch_gt.save(name_patch_gt)
                one_refine.save(name_refine)
                index_save = index_save + 1

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val * (iter_per_epoch - i))))

        # Log progress
        niter = epoch * iter_per_epoch + i + 1
        if (i + 1) % 100 == 0 or (i + 1) == iter_per_epoch:
            temp_mean_loss = np.asarray(temp_series_loss).mean()
            temp_mean_l_patch = np.asarray(temp_series_l_patch).mean()
            temp_mean_l_patch_dx = np.asarray(temp_series_l_patch_dx).mean()
            temp_mean_l_patch_dy = np.asarray(temp_series_l_patch_dy).mean()
            temp_series_l_patch, temp_series_l_patch_dx, temp_series_l_patch_dy = [], [], []
            series_loss.append(temp_mean_loss)
            series_l_patch.append(temp_mean_l_patch)
            series_l_patch_dx.append(temp_mean_l_patch_dx)
            series_l_patch_dy.append(temp_mean_l_patch_dy)

            # Print to console
            print('  ')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'ETA {eta}\t    '
                  .format(epoch, i + 1, iter_per_epoch, batch_time=batch_time, eta=eta), end='')

            print('|| current iter: ' + str(current_iter).zfill(7) + ' || ', end='')
            print("loss: {:10.7f}, ".format(temp_mean_loss), end='')
            print("l_patch: {:10.7f}, ".format(temp_mean_l_patch), end='')
            print("l_patch_dx: {:10.7f}, ".format(temp_mean_l_patch_dx), end='')
            print("l_patch_dy: {:10.7f}, ".format(temp_mean_l_patch_dy), end='')

        if (current_iter % save_period == 0) and MODE == 'TRAIN':
            model_name = model_path + "/" + 'iter' + str(current_iter).zfill(7) + ".pth"
            # save model
            print('SAVE MODEL:' + model_path)
            torch.save(model.state_dict(), model_name)

    return current_iter, series_loss, series_l_patch, series_l_patch_dx, series_l_patch_dy
