
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import numpy as np
import torchvision.utils as vutils
from torchvision import transforms
import torch
import torchvision
from datetime import datetime
from gan import *





'''
print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_num > 0) else "cpu")

generator = Generator(gen_input)
generator.load_state_dict(torch.load('res/out/generator(28.04.2021 17-26-04).pt'))
generator.eval()
generator.to(device)

fixed_noise = torch.randn(1, gen_input, 1, 1).to(device=device)
generated_samples = generator(fixed_noise)

generated_samples = generated_samples.cpu().detach()

mean = 0.5
std = 0.5
unnorm = transforms.Normalize((-mean / std, ), (1.0 / std, ))
generated_samples = unnorm(generated_samples)
for el in generated_samples:
    gen_nodule = transforms.ToPILImage()(el)
    gen_nodule.show()

plt.figure(figsize=(5, 5))
plt.axis("off")
plt.title("Сгенерированный образец")
plt.imshow(np.transpose(vutils.make_grid(generated_samples, normalize=False).cpu()))
plt.show()
'''
