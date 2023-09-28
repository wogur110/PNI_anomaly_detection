import os
import torch
import numpy as np
import random
import glob
from PIL import Image, ImageStat
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def listData(input_data_path):
    target_patch_mid = os.path.join(input_data_path, "*_pred.png")
    target_patch_gt = os.path.join(input_data_path, "*_gt.png")
    target_image_gt = os.path.join(input_data_path, "image*.png")

    list_patch_mid = glob.glob(target_patch_mid)
    list_patch_gt = glob.glob(target_patch_gt)
    list_image_gt = glob.glob(target_image_gt)

    list_patch_mid.sort()
    list_patch_gt.sort()
    list_image_gt.sort()

    return list_patch_mid, list_patch_gt, list_image_gt

def getData(list_patch_mid, list_patch_gt, list_image_gt, batch_size, MODE='TRAIN', input_size=[320,320]):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    if MODE == 'TRAIN':
        shuffle = True
        transformed_data = transforms.Compose([
            Scale(input_size),
            RandomHorizontalFlip(),
            RandomRotate(5),
            ToTensor_with_RandomZoom(ratio=1.00),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, ),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            ])

    if MODE == 'EVAL':
        shuffle = False
        transformed_data = transforms.Compose([
            Scale(input_size),
            ToTensor_with_RandomZoom(ratio=1.00),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            ])

    transformed_data = depthDatasetMemoryTrain(list_patch_mid, list_patch_gt, list_image_gt, transform=transformed_data)

    return DataLoader(transformed_data, batch_size, shuffle=shuffle), len(list_patch_mid)

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        if not _is_pil_image(image_in): raise TypeError('img should be PIL Image. Got {}'.format(type(image_in)))
        if not _is_pil_image(patch_mid): raise TypeError('img should be PIL Image. Got {}'.format(type(patch_mid)))
        if not _is_pil_image(image_gt): raise TypeError('img should be PIL Image. Got {}'.format(type(image_gt)))
        if not _is_pil_image(patch_gt): raise TypeError('img should be PIL Image. Got {}'.format(type(patch_gt)))

        if random.random() < 0.5:
            image_in = image_in.transpose(Image.FLIP_LEFT_RIGHT)
            patch_mid = patch_mid.transpose(Image.FLIP_LEFT_RIGHT)
            image_gt = image_gt.transpose(Image.FLIP_LEFT_RIGHT)
            patch_gt = patch_gt.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        if not _is_pil_image(image_in): raise TypeError('img should be PIL Image. Got {}'.format(type(image_in)))
        if not _is_pil_image(patch_mid): raise TypeError('img should be PIL Image. Got {}'.format(type(patch_mid)))
        if not _is_pil_image(image_gt): raise TypeError('img should be PIL Image. Got {}'.format(type(image_gt)))
        if not _is_pil_image(patch_gt): raise TypeError('img should be PIL Image. Got {}'.format(type(patch_gt)))
        if random.random() < self.probability:
            rand_index = random.randint(0, len(self.indices) - 1)
            image_in = np.asarray(image_in)
            image_in = Image.fromarray(image_in[...,list(self.indices[rand_index])])
            image_gt = np.asarray(image_gt)
            image_gt = Image.fromarray(image_gt[...,list(self.indices[rand_index])])
        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

def loadZipToMem(zip_file, csv_name):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')

    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    train = list((row.split(',') for row in (data[csv_name]).decode("utf-8").split('\n') if len(row) > 0))
    #from sklearn.utils import shuffle
    #train = shuffle(train, random_state=0)

    print('Loaded ({0}) data.'.format(len(train)))
    return data, train

class depthDatasetMemoryTrain(Dataset):
    def __init__(self, list_patch_mid, list_patch_gt, list_image_gt, transform=None):
        self.list_patch_mid, self.list_patch_gt, self.list_image_gt = list_patch_mid, list_patch_gt, list_image_gt
        self.transform = transform
        self.maxDepth = 1000.0

    def __getitem__(self, idx):
        image_in = Image.open(self.list_image_gt[idx].replace('\\', '/'))
        patch_mid = Image.open(self.list_patch_mid[idx].replace('\\', '/'))
        image_gt = Image.open(self.list_image_gt[idx].replace('\\', '/'))
        patch_gt = Image.open(self.list_patch_gt[idx].replace('\\', '/'))

        sample = {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.list_patch_mid)

class ToTensor_with_RandomZoom(object):
    def __init__(self, ratio=1):
        self.ratio = ratio

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        original_size = image_in.size
        applied_zoom = random.uniform(1, self.ratio)

        image_in, patch_mid, image_gt, patch_gt = self.zoom(image_in, patch_mid, image_gt, patch_gt, applied_zoom)
        image_in, patch_mid, image_gt, patch_gt = self.randomCrop(image_in, patch_mid, image_gt, patch_gt, original_size)

        image_in = self.to_tensor(image_in)
        image_gt = self.to_tensor(image_gt)
        if image_in.shape[0] == 1:
            image_in = image_in.repeat(3,1,1)
        if image_gt.shape[0] == 1:
            image_gt = image_gt.repeat(3,1,1)
        patch_mid = self.to_tensor(patch_mid).float()
        patch_gt = self.to_tensor(patch_gt).float()

        if patch_mid.shape[0] == 3:
            patch_mid = torch.mean(patch_mid, dim=0, keepdim=True)
        if patch_gt.shape[0] == 3:
            patch_gt = torch.mean(patch_gt, dim=0, keepdim=True)

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

    def zoom(self, image_in, patch_mid, image_gt, patch_gt, applied_zoom):
        w1, h1 = image_in.size
        w2 = round(w1 * applied_zoom)
        h2 = round(h1 * applied_zoom)

        image_in = image_in.resize((w2, h2), Image.BICUBIC)
        patch_mid = patch_mid.resize((w2, h2), Image.BICUBIC)
        image_gt = image_gt.resize((w2, h2), Image.BICUBIC)
        patch_gt = patch_gt.resize((w2, h2), Image.BICUBIC)

        return image_in, patch_mid, image_gt, patch_gt

    def randomCrop(self, image_in, patch_mid, image_gt, patch_gt, size):
        w1, h1 = size
        w2, h2 = image_in.size

        if w1 == w2 and h1 == h2:
            return image_in, patch_mid, image_gt, patch_gt

        x = round(random.uniform(0, w2 - w1) - 0.5)
        y = round(random.uniform(0, h2 - h1) - 0.5)

        image_in = image_in.crop((x, y, x + w1, y + h1))
        patch_mid = patch_mid.crop((x, y, x + w1, y + h1))
        image_gt = image_gt.crop((x, y, x + w1, y + h1))
        patch_gt = patch_gt.crop((x, y, x + w1, y + h1))

        return image_in, patch_mid, image_gt, patch_gt

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False)).float().div(65535)
        elif pic.mode == 'L':
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes())).float().div(255)
        elif pic.mode == 'RGB':
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes())).float().div(255)
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img

class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        image_in = self.changeScale(image_in, self.size)
        patch_mid = self.changeScale(patch_mid, self.size, Image.NEAREST)
        image_gt = self.changeScale(image_gt, self.size)
        patch_gt = self.changeScale(patch_gt, self.size, Image.NEAREST)

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        #if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        #    raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        #image_original = image.clone()
        image_in = self.normalize(image_in, self.mean, self.std)
        image_gt = self.normalize(image_gt, self.mean, self.std)

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']
        mean_patch_mid = round(ImageStat.Stat(patch_mid).mean[0] * 255)
        mean_patch_gt = round(ImageStat.Stat(patch_gt).mean[0] * 255)

        applied_angle = random.uniform(-self.angle, self.angle)

        if image_in.mode == 'L':
            image_in = image_in.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(255))
        else:
            image_in = image_in.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(255,255,255))
        patch_mid = patch_mid.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(mean_patch_mid))

        if image_gt.mode == 'L':
            image_gt = image_gt.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(255))
        else:
            image_gt = image_gt.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(255,255,255))
        patch_gt = patch_gt.rotate(applied_angle, resample=Image.BILINEAR, fillcolor=(mean_patch_gt))

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

# TODO: Lighting 재검토
class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']
        if self.alphastd == 0:
            return image_in

        alpha = image_in.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image_in).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        image_in = image_in.add(rgb.view(3, 1, 1).expand_as(image_in))

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        if self.transforms is None:
            return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}
        order = torch.randperm(len(self.transforms))
        for i in order:
            image_in = self.transforms[i](image_in)

        return {'image_in': image_in, 'patch_mid': patch_mid, 'image_gt': image_gt, 'patch_gt': patch_gt}

class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.Brightness = Brightness(brightness)
        self.Contrast = Brightness(contrast)
        self.Saturation = Brightness(saturation)

    def __call__(self, sample):
        sample = self.Brightness(sample)
        sample = self.Contrast(sample)
        sample = self.Saturation(sample)
        return sample

class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        image_in_gs = Grayscale()(image_in)
        image_gt_gs = Grayscale()(image_gt)
        alpha = random.uniform(-self.var, self.var)

        return {
            'image_in': image_in.lerp(image_in_gs, alpha),
            'patch_mid': patch_mid,
            'image_gt': image_gt.lerp(image_gt_gs, alpha),
            'patch_gt': patch_gt}

class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']
        image_in_gs = image_in.new().resize_as_(image_in).zero_()
        image_gt_gs = image_gt.new().resize_as_(image_gt).zero_()
        alpha = random.uniform(-self.var, self.var)

        return {
            'image_in': image_in.lerp(image_in_gs, alpha),
            'patch_mid': patch_mid,
            'image_gt': image_gt.lerp(image_gt_gs, alpha),
            'patch_gt': patch_gt}

class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        image_in, patch_mid, image_gt, patch_gt \
            = sample['image_in'], sample['patch_mid'], sample['image_gt'], sample['patch_gt']

        image_in_gs = Grayscale()(image_in)
        image_gt_gs = Grayscale()(image_gt)
        image_in_gs.fill_(image_in_gs.mean())
        image_gt_gs.fill_(image_gt_gs.mean())
        alpha = random.uniform(-self.var, self.var)

        return {
            'image_in': image_in.lerp(image_in_gs, alpha),
            'patch_mid': patch_mid,
            'image_gt': image_gt.lerp(image_gt_gs, alpha),
            'patch_gt': patch_gt}

class Grayscale(object):

    def __call__(self, img):
        gs = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return gs