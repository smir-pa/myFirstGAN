
from torch import nn
from params import *


# Инициализация весов генератора/дискриминатора
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Генератор
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, gen_fms * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_fms * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_fms * 8, gen_fms * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_fms * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_fms * 4, gen_fms * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_fms * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_fms * 2, gen_fms, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_fms),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_fms, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, dis_fms, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_fms, dis_fms * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_fms * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_fms * 2, dis_fms * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_fms * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_fms * 4, dis_fms * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_fms * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_fms * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
