
import os
import shutil
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from params import *
from gan import Discriminator, Generator, weights_init
from datetime import datetime


if __name__ == '__main__':
    t0 = datetime.now()
    if not os.path.exists('res/out'):
        os.mkdir('res/out')

    # Для повторного воспроизведения эксперимента
    torch.manual_seed(42)

    # Загрузка и предобработка датасета
    if channels == 3:
        dataset = dset.ImageFolder(path,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        dataset = dset.ImageFolder(path,
                                   transform=transforms.Compose([
                                       transforms.Grayscale(1),
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, ), (0.5, )),
                                   ]))

    # Создание загрузчика
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Выбор устройства (GPU/CPU)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_num > 0) else "cpu")

    # Создание генератора
    generator = Generator(gen_input).to(device)

    # Включение нескольких GPU, если возможно
    if (device.type == 'cuda') and (gpu_num > 1):
        generator = nn.DataParallel(generator, list(range(gpu_num)))

    # Инициализация весов генератора
    generator.apply(weights_init)
    print(generator)

    # Создание дискриминатора
    discriminator = Discriminator().to(device)

    # Включение нескольких GPU, если возможно
    if (device.type == 'cuda') and (gpu_num > 1):
        discriminator = nn.DataParallel(discriminator, list(range(gpu_num)))

    # Инициализация весов генератора
    discriminator.apply(weights_init)
    print(discriminator)

    # Функция потерь - Бинарная перекрестная энтропия
    loss_func = nn.BCELoss()

    # Создание вектора шума для генератора
    fixed_noise = torch.randn(batch_size, gen_input, 1, 1, device=device)

    # Метки классов
    real_label = 1.
    fake_label = 0.

    # Оптимизаторы Adam (стохастический градиентный спуск) для дискриминатора и генератора
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=dis_l_rate, betas=(beta1, beta2))
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_l_rate, betas=(beta1, beta2))

    # Цикл обучения

    # Для отслеживания прогресса
    img_list = []
    gen_losses = []
    dis_losses = []
    iter_num = 0

    print("#"*16 + " Начало цикла " + "#"*16)
    # Для каждой эпохи
    for epoch in range(epochs):
        # Для каждой части выборки в загрузчике
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Обновить D: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Тренировка со всеми реальными образцами
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Пропустить реальные через D
            output = discriminator(real_cpu).view(-1)
            # Высчитать потери на всех реальных
            dis_err_real = loss_func(output, label)
            # Высчитать градиенты для D в обратном проходе
            dis_err_real.backward()
            dis_x = output.mean().item()

            # Тренировка со всеми поддельными образцами
            # Генерация пачки векторов шума
            noise = torch.randn(b_size, gen_input, 1, 1, device=device)
            # Генерация поддельных с помощью G
            fake = generator(noise)
            label.fill_(fake_label)
            # Классифицировать поддельные с помощью D
            output = discriminator(fake.detach()).view(-1)
            # Высчитать потери D на поддельных
            dis_err_fake = loss_func(output, label)
            # Высчитать градиенты
            dis_err_fake.backward()
            dis_gen_z1 = output.mean().item()
            # Сложить градиенты реальных и поддельных
            dis_err = dis_err_real + dis_err_fake
            # Обновить D
            dis_optimizer.step()

            ############################
            # (2) Обновить G: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)
            # Т.к. D только что обновлен, совершить еще один обратный проход поддельных через D
            output = discriminator(fake).view(-1)
            # Высчитать потери G, основываясь на этом выводе
            gen_err = loss_func(output, label)
            # Высчитать градиенты для G
            gen_err.backward()
            dis_gen_z2 = output.mean().item()
            # Обновить G
            gen_optimizer.step()

            # Вывод статистики обучения
            if i % 50 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, epochs, dis_err.item(), gen_err.item(), dis_x, dis_gen_z1, dis_gen_z2))

            # Сохранение потерь для дальнейшего построения
            gen_losses.append(gen_err.item())
            dis_losses.append(dis_err.item())

            # Проверка того, как генератор справляется, сохраняя вывод G
            if (iter_num % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iter_num += 1

    t1 = datetime.now() - t0
    t_save = datetime.now().strftime('(%d.%m.%Y %H-%M-%S)')
    print("Времени прошло, с:", t1.seconds)

    # Построение графика изменения потерь
    plt.figure(figsize=(10, 5))
    plt.title("Потери генератора и дискриминатора во время обучения")
    plt.plot(gen_losses, label="G")
    plt.plot(dis_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('res/out/loss' + t_save + '.png')
    plt.show()

    '''# Анимация изменений
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())'''

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
    plt.savefig('res/out/real_vs_fake' + t_save + '.png')
    plt.show()

    # Сохранение весов модели генератора
    torch.save(generator.state_dict(), 'res/out/generator' + t_save + '.pt')

    # Параметры сохраненной модели
    shutil.copy('params.py', 'res/out/params' + t_save + '.txt')
