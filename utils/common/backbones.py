import torch

def Backbone(backbone_name):
    if backbone_name == 'R152' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    elif backbone_name == 'R101' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    elif backbone_name == 'R18' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    elif backbone_name == 'R34' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    elif backbone_name == 'R50' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    elif backbone_name == 'WR50' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    elif backbone_name == 'WR101' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
    elif backbone_name == 'RNX101' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
    elif backbone_name == 'DN201' :
        return torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)