
import os
import shutil
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from context_params import *
from context_gan import Discriminator, Generator, weights_init_normal, ImageDataset
from datetime import datetime


if __name__ == '__main__':
    t0 = datetime.now()
    if not os.path.exists('res/context_out'):
        os.mkdir('res/context_out')

    # Для повторного воспроизведения эксперимента
    torch.manual_seed(42)

    # Загрузчик и предобработка датасета
    if channels == 3:
        transforms_ = [
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        dataloader = DataLoader(
            ImageDataset(path, transforms_=transforms_),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
        )
        test_dataloader = DataLoader(
            ImageDataset(path, transforms_=transforms_, mode="val"),
            batch_size=12,
            shuffle=True,
            num_workers=1,
        )
    else:
        transforms_ = [
            transforms.Grayscale(1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ]
        img_dset = ImageDataset(path, transforms_=transforms_)
        dataloader = DataLoader(
            img_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
        )
        test_dataloader = DataLoader(
            ImageDataset(path, transforms_=transforms_, mode="val"),
            batch_size=16,
            shuffle=True,
            num_workers=1,
        )

    # Выбор устройства (GPU/CPU)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_num > 0) else "cpu")

    # Создание генератора
    generator = Generator(channels=channels).to(device)

    # Использование предобученной модели генератора
    if load_pretrained_models:
        generator.load_state_dict(torch.load(gen_model_path))
        print("Используется предобученная модель генератора...")

    # Включение нескольких GPU, если возможно
    if (device.type == 'cuda') and (gpu_num > 1):
        generator = nn.DataParallel(generator, list(range(gpu_num)))

    # Инициализация весов генератора
    generator.apply(weights_init_normal)
    print(generator)

    # Создание дискриминатора
    discriminator = Discriminator(channels=channels).to(device)

    # Использование предобученной модели дискриминатора
    if load_pretrained_models:
        discriminator.load_state_dict(torch.load(dis_model_path))
        print("Используется предобученная модель дискриминатора")

    # Включение нескольких GPU, если возможно
    if (device.type == 'cuda') and (gpu_num > 1):
        discriminator = nn.DataParallel(discriminator, list(range(gpu_num)))

    # Инициализация весов генератора
    discriminator.apply(weights_init_normal)
    print(discriminator)

    # Функции потерь
    adversarial_loss = nn.MSELoss()
    pixelwise_loss = nn.L1Loss()

    # Оптимизаторы Adam (стохастический градиентный спуск) для дискриминатора и генератора
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=dis_l_rate, betas=(beta1, beta2))
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_l_rate, betas=(beta1, beta2))

    # Настройка тензора
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Цикл обучения

    # Для отслеживания прогресса
    img_list = []
    gen_losses = []
    gen_pix_losses = []
    dis_losses = []
    counter = []

    print("#"*16 + " Начало цикла " + "#"*16)
    # Для каждой эпохи
    for epoch in range(epochs):
        gen_loss, gen_pix_loss, dis_loss = 0, 0, 0
        tqdm_bar = tqdm(dataloader, desc=f'Training Epoch {epoch} ', total=int(len(dataloader)))

        # Для каждой части выборки в загрузчике
        for i, (imgs, masked_imgs, masked_parts) in enumerate(tqdm_bar):
            # Метки классов
            real = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
            # Настройка входа
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))

            ############################
            # (1) Тренировка генератора
            ###########################
            gen_optimizer.zero_grad()
            # Генерация пачки изображений
            gen_parts = generator(masked_imgs)
            # Вычисление потерь
            g_adv = adversarial_loss(discriminator(gen_parts), real)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            # Общие потери
            g_loss = 0.001 * g_adv + 0.999 * g_pixel
            # Высчитать градиенты для G
            g_loss.backward()
            # Обновить G
            gen_optimizer.step()

            ############################
            # (2) Тренировка дискриминатора
            ###########################
            dis_optimizer.zero_grad()
            # Способность дискриминатора классифицировать образцы
            real_loss = adversarial_loss(discriminator(masked_parts), real)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            # Высчитать градиенты для D
            d_loss.backward()
            # Обновить D
            dis_optimizer.step()

            # Сохранение потерь для дальнейшего использования
            gen_loss += g_adv.item()
            gen_pix_loss += g_pixel.item()
            gen_losses.append(g_adv.item())
            gen_pix_losses.append(g_pixel.item())
            dis_loss += d_loss.item()
            dis_losses.append(d_loss.item())
            counter.append(i * batch_size + imgs.size(0) + epoch * len(dataloader.dataset))
            tqdm_bar.set_postfix(gen_adv_loss=gen_loss / (i + 1), gen_pixel_loss=gen_pix_loss / (i + 1),
                                 disc_loss=dis_loss / (i + 1))

            # Generate sample at sample interval
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                samples, masked_samples, i = next(iter(test_dataloader))
                samples = Variable(samples.type(Tensor))
                masked_samples = Variable(masked_samples.type(Tensor))
                i = i[0].item()  # Левый верхний пиксель маски
                # Генерация изображения
                gen_mask = generator(masked_samples)
                filled_samples = masked_samples.clone()
                filled_samples[:, :, i: i + mask_size, i: i + mask_size] = gen_mask
                # Сохранение
                sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
                img_list.append(vutils.make_grid(filled_samples.detach().cpu(), padding=2, normalize=True))
                save_image(sample, "res/images/%d.jpg" % batches_done, nrow=6, normalize=True)

    t1 = datetime.now() - t0
    t_save = '_context' + datetime.now().strftime('(%d.%m.%Y %H-%M-%S)')
    print("Времени прошло, с:", t1.seconds)

    # Построение графика изменения потерь
    plt.figure(figsize=(10, 5))
    plt.title("Потери генератора и дискриминатора во время обучения")
    plt.plot(gen_losses, label="G")
    plt.plot(dis_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('res/context_out/loss' + t_save + '.png')
    plt.show()

    # Пачка реальных изображений из загрузчика
    real_batch = next(iter(dataloader))

    # Вывод реальных
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Реальные")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                             padding=2, normalize=True).cpu(), (1, 2, 0)))

    # Вывод поддельных из последней эпохи
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Поддельные (совсем как настоящие)")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig('res/context_out/real_vs_fake' + t_save + '.png')
    plt.show()

    # Сохранение весов модели генератора и дискриминатора
    torch.save(generator.state_dict(), 'res/context_out/generator' + t_save + '.pt')
    torch.save(discriminator.state_dict(), 'res/context_out/discriminator' + t_save + '.pt')

    # Параметры сохраненной модели
    shutil.copy('context_params.py', 'res/context_out/params' + t_save + '.txt')
