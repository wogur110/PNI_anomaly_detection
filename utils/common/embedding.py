import torch
from torch.nn import functional as F

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

def generate_embedding_features(args, features, patch_maker):
    features = [features[layer] for layer in args.layer_index]
    '''
    # m = torch.nn.AvgPool2d(kernel_size = args.patchsize, stride = 1, padding = 0, count_include_pad = False)
    # pad = ((args.patchsize - 1) // 2, (args.patchsize - 1) // 2, (args.patchsize - 1) // 2, (args.patchsize - 1) // 2)
    # for i in range(len(features)) :
    #     _features = features[i]
    #     _features = F.pad(_features, pad, "reflect")
    #     features[i] = m(_features)
    m = torch.nn.AvgPool2d(kernel_size = args.patchsize, stride = 1, padding = (args.patchsize - 1) // 2)
    features = [  
        m(x) for x in features
    ]
    
    patch_shapes = []
    for i in range(len(features)):
        patch_shapes.append(features[i].shape[-2:])
    
    ref_num_patches = patch_shapes[0]
    
    for i in range(len(features)) :
        _features = features[i]
        patch_dims = patch_shapes[i]
        if i > 0 :
            _features = F.interpolate(
                _features,
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
        _features = _features.permute(0, -2, -1, 1) # N x W x H x E
        _features = _features.reshape(-1, 1, _features.shape[-1]) #(N x W x H) x 1 x E
        _features = F.adaptive_avg_pool1d(_features, args.pretrain_embed_dimension).squeeze(1) #(N x W x H) x ED
        features[i] = _features
        
    features = torch.stack(features, dim=1) # (N x W x H) x 2 x ED
            
    '''     
    features = [
        patch_maker.patchify(x, return_spatial_info=True) for x in features
    ]
    
    patch_shapes = [x[1] for x in features]
    features = [x[0] for x in features]
    ref_num_patches = patch_shapes[0]
    
    for i in range(1, len(features)):
        _features = features[i]
        patch_dims = patch_shapes[i]

        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
        )
        _features = _features.permute(0, -3, -2, -1, 1, 2)
        perm_base_shape = _features.shape
        _features = _features.reshape(-1, *_features.shape[-2:])
        _features = F.interpolate(
            _features.unsqueeze(1),
            size=(ref_num_patches[0], ref_num_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        _features = _features.squeeze(1)
        _features = _features.reshape(
            *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
        )
        _features = _features.permute(0, -2, -1, 1, 2, 3)
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
        features[i] = _features
    features = [x.reshape(-1, *x.shape[-3:]) for x in features]
    # preprocessing
    _features = []
    for feature in features : 
        feature = feature.reshape(len(feature), 1, -1)
        _features.append(F.adaptive_avg_pool1d(feature, args.pretrain_embed_dimension).squeeze(1))
        
    features = torch.stack(_features, dim=1)
    
    # preadapt aggregator
    _features = features.reshape(len(features), 1, -1) #(N x W x H) x 1 x (2 x ED)
    _features = F.adaptive_avg_pool1d(_features, args.target_embed_dimension) #(N x W x H) x 1 x (TED)
    features = _features.reshape(len(features), -1) #(N x W x H) x (TED)
    
    return features, ref_num_patches