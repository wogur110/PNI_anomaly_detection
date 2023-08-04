from torchvision import transforms
import torch

#imagenet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def Transform(resize, imagesize) :
    """
    Default transform from Image to normalized tensor
    Args:
        resize (int): Resize shape
        imagesize (int): CenterCrop shape
    """
    transform = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.CenterCrop(imagesize),
                        transforms.ToTensor(),                        
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform

def GT_Transform(resize, imagesize) :
    """
    Default transform from ground truth image to tensor (not normalize)
    Args:
        resize (int): Resize shape
        imagesize (int): CenterCrop shape
    """
    transform = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.CenterCrop(imagesize),
                        transforms.ToTensor()])
    return transform

def INV_Normalize() :
    """
    Inverse normalize from normalized tensor
    """
    transform = transforms.Normalize(mean = - torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD), std = 1 / torch.tensor(IMAGENET_STD))
    return transform