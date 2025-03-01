import torch
from torchvision import transforms
from utilities import load_config


class SiameseTransform:
    def __init__(self, mean, std, mode, DATA_MODES):
        self.mode = mode
        
        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        train = mode == 'train'
        self.config = load_config("configs/", "config_transform.yaml")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=self.config['transform']['horizontalflip_prop']) if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(degrees=self.config['transform']['rotation_degrees']) if train else transforms.Lambda(lambda x: x),
            transforms.RandomPerspective(distortion_scale=self.config['transform']['perspective_distortion'], p=self.config['transform']['perspective_prop']) if train else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transform(img)