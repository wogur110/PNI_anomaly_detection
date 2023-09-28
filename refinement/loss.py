import torch
import torch.nn.functional as F
import numpy as np

def compute_3losses(depth_pred_for_loss, depth_gt_for_loss, n=1):
    l_depth, l_depth_dx, l_depth_dy = loss_for_derivative_mini(depth_pred_for_loss, depth_gt_for_loss, n=n, one_metric=False)

    return l_depth, l_depth_dx, l_depth_dy

def loss_for_derivative_mini(pred, gt, n=1, one_metric=False):
    # estimate loss of derivative
    #   d, dx, dy, d_norm, dx2, dxy, dy2, dx_norm, dy_norm

    pred_dx = gradient_dx(pred)
    pred_dy = gradient_dy(pred)

    gt_dx = gradient_dx(gt)
    gt_dy = gradient_dy(gt)

    l_depth = loss_Ln_by_channel(pred, gt, n=n)
    l_depth_dx = loss_Ln_by_channel(pred_dx, gt_dx, n=n)
    l_depth_dy = loss_Ln_by_channel(pred_dy, gt_dy, n=n)

    if one_metric == False:
        return l_depth, l_depth_dx, l_depth_dy
    elif one_metric == True:
        return (l_depth + l_depth_dx + l_depth_dy)/3

def gradient_dx(img):
    (_, channel, height, width) = img.size()

    kernel_dx = np.zeros((channel, 1, 1, 2))
    kernel_dx[:, :, :, 0] =  1
    kernel_dx[:, :, :, 1] = -1
    kernel_dx = torch.tensor(kernel_dx, device=img.device).float()
    img2kernel = F.conv2d(F.pad(img, pad=(0,1,0,0), mode='replicate'), kernel_dx, padding=0, groups=channel)

    return img2kernel

def gradient_dy(img):
    (_, channel, height, width) = img.size()

    kernel_dy = np.zeros((channel, 1, 2, 1))
    kernel_dy[:, :, 0, :] =  1
    kernel_dy[:, :, 1, :] = -1
    kernel_dy = torch.tensor(kernel_dy, device=img.device).float()

    img2kernel = F.conv2d(F.pad(img, pad=(0,0,0,1), mode='replicate'), kernel_dy, padding=0, groups=channel)

    return img2kernel

def loss_Ln_by_channel(pred, gt, n=2):

    loss = torch.mean(torch.abs(torch.pow(pred-gt,n)), dim=[1, 2, 3])

    return loss
