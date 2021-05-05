
import glob
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from context_params import *


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, image_size=image_size, mask_size=mask_size, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.image_size = image_size
        self.mask_size = mask_size
        self.mode = mode
        if mode == "train":
            self.files = sorted(glob.glob("%s/**/*.jpg" % root, recursive=True))
        else:
            self.files = [root]

    # Наложение маски в случайной области
    def apply_random_mask(self, img):
        y1, x1 = np.random.randint(0, self.image_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    # Наложение маски по центру изображения
    def apply_center_mask(self, img):
        i = (self.image_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # Во время обучения используется случайное наложение
            masked_img, aux = self.apply_random_mask(img)
        else:
            # Для тестовых данных расположение маски по центру
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


# Инициализация весов генератора/дискриминатора
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# Генератор
class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels, gen_fms, normalize=False),
            *downsample(gen_fms, gen_fms),
            *downsample(gen_fms, 2*gen_fms),
            *downsample(2*gen_fms, 4*gen_fms),
            *downsample(4*gen_fms, 8*gen_fms),
            nn.Conv2d(8*gen_fms, 4000, 1),
            *upsample(4000, 8*gen_fms),
            *upsample(8*gen_fms, 4*gen_fms),
            *upsample(4*gen_fms, 2*gen_fms),
            *upsample(2*gen_fms, gen_fms),
            nn.Conv2d(gen_fms, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Дискриминатор

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(dis_fms, 2, False), (2*dis_fms, 2, True),
                                               (4*dis_fms, 2, True), (8*dis_fms, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
