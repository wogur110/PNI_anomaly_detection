import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import refinement.timm as timm

def create_model(model_name, decoder_scale=1024):
    # Create model
    if model_name == 'densenet161_early_fusion':
        model = Model_DenseNet161_Early_Fusion(decoder_scale).cuda()

    return model

class Model_DenseNet161_Early_Fusion(nn.Module):
    def __init__(self, decoder_scale = 512):
        super(Model_DenseNet161_Early_Fusion, self).__init__()
        self.encoder1 = Encoder_DenseNet161()
        self.encoder2 = Encoder_DenseNet161()
        self.decoder = Decoder_DenseNet161(decoder_scale=decoder_scale)

    def forward(self, image, patch=None):
        if patch == None:
            patch = image * 0
        features_encoder_early1 = self.encoder1.original_model.forward_densenet_get_early_features(image)
        features_encoder_early2 = self.encoder2.original_model.forward_densenet_get_early_features(patch)
        features_encoder_early = features_encoder_early1/2 + features_encoder_early2/2
        features_encoder = self.encoder1.original_model.forward_densenet_get_late_features(features_encoder_early, pool=False)
        return self.decoder(features_encoder)

class Encoder_DenseNet161(nn.Module):
    def __init__(self):
        super(Encoder_DenseNet161, self).__init__()
        self.original_model = timm.create_model("densenet161", pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class Decoder_DenseNet161(nn.Module):
    def __init__(self, decoder_scale = 512, num_upsample = 5, num_channels_final = 1):
        super(Decoder_DenseNet161, self).__init__()

        num_channels_in = 2208
        num_channels_out = decoder_scale
        self.features = nn.Sequential(OrderedDict([
            ('decoder_conv0', nn.Conv2d(num_channels_in, num_channels_out, kernel_size=1, stride=1)),
        ]))

        for index_upsample in range(5):
            if num_upsample < index_upsample+1:
                num_channels_in = num_channels_out
                num_channels_out = num_channels_in
                block = SameSample_simple(skip_input=num_channels_in, output_features=num_channels_out)
                self.features.add_module('decoder_same%d' % (index_upsample+1), block)
            else:
                num_channels_in = num_channels_out
                num_channels_out = num_channels_in // 2
                block = UpSample_simple(skip_input=num_channels_in, output_features=num_channels_out)
                self.features.add_module('decoder_up%d' % (index_upsample+1), block)

        num_channels_in = num_channels_out
        num_channels_out = num_channels_final
        block = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=1, stride=1)
        self.features.add_module('decoder_conv_final', block)

    def forward(self, features_encoder):
        features_decoder = self.features(features_encoder)
        return features_decoder

class UpSample_simple(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample_simple, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=1, stride=1, padding=0)
        self.reluA = nn.ReLU()
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.reluB = nn.ReLU()

    def forward(self, x):
        up_x = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
        return self.reluB( self.convB( self.reluA(self.convA( up_x ) ) )  )

class SameSample_simple(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(SameSample_simple, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=1, stride=1, padding=0)
        self.reluA = nn.ReLU()
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.reluB = nn.ReLU()

    def forward(self, x):
        return self.reluB( self.convB( self.reluA(self.convA( x ) ) )  )