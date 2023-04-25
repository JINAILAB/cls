import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.139, 0, 0),
        std=(0.073, 1, 1),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        channel1 = None
    ):
        trans = [transforms.Resize(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        if channel1:
            trans.extend( [transforms.Grayscale(num_output_channels=1)])
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=300,
        mean=(0.139, 0, 0),
        std=(0.073,1,1),
        interpolation=InterpolationMode.BILINEAR,
        channel1=None
    ):
        trans = [transforms.PILToTensor(),
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size)]
        
        if channel1:
            trans.extend( [transforms.Grayscale(num_output_channels=1)])
        trans.extend([transforms.ConvertImageDtype(torch.float),
                      transforms.Normalize(mean=mean, std=std)])

        self.transforms = transforms.Compose(trans)
        

    def __call__(self, img):
        return self.transforms(img)