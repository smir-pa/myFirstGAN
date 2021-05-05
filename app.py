import os

from PIL import Image as PILimg
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable as Var
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from gan import Generator
from context_gan import Generator as ContextGenerator, ImageDataset
from params import *
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Style


class App(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Image Modifier")
        self.pack(fill=BOTH, expand=1)
        self.centerWindow()
        self.parent.resizable(False, False)
        Style().configure("TButton", padding=(0, 5, 0, 5), font='serif 10')

        self.columnconfigure(0, pad=3)
        self.columnconfigure(1, pad=3)
        self.columnconfigure(2, pad=3)
        self.columnconfigure(3, pad=3)
        self.columnconfigure(4, pad=3)
        self.columnconfigure(5, pad=3)

        self.rowconfigure(0, pad=3)
        self.rowconfigure(1, pad=3)
        self.rowconfigure(2, pad=3)
        self.rowconfigure(3, pad=3)
        self.rowconfigure(4, pad=3)

        self.lbl_file = Label(self, text="Выберите файл для модификации:")
        self.lbl_file.grid(row=0, column=0, sticky=W, padx=3)
        self.line_file = Entry(self)
        self.line_file.grid(row=1, column=0, sticky=W+E, columnspan=4, padx=5)
        self.btn_search_file = Button(self, text="Обзор...", command=lambda: self.load('image'))
        self.btn_search_file.grid(row=1, column=4, sticky=W+E)
        self.btn_show_file = Button(self, text="Просмотр", command=lambda: self.show_graph('original'))
        self.btn_show_file.grid(row=1, column=5, sticky=W+E)

        self.lbl_model = Label(self, text="Выберите модель для генерации:")
        self.lbl_model.grid(row=2, column=0, sticky=W, padx=3)
        self.line_model = Entry(self)
        self.line_model.grid(row=3, column=0, sticky=W+E, columnspan=4, padx=5)
        self.btn_search_model = Button(self, text="Обзор...", command=lambda: self.load('model'))
        self.btn_search_model.grid(row=3, column=4, sticky=W+E)
        self.btn_show_gen = Button(self, text="Генерация", command=lambda: self.show_graph('generated'))
        self.btn_show_gen.grid(row=3, column=5, sticky=W+E)

        self.lbl_coords = Label(self, text="Координаты вставки:")
        self.lbl_coords.grid(row=4, column=0, sticky=W, padx=3)
        self.lbl_x = Label(self, text="X:")
        self.lbl_x.grid(row=5, column=0, sticky=W, padx=3)
        self.spin_x = Spinbox(self, width=8, from_=0, to=10000)
        self.spin_x.grid(row=5, column=1, sticky=W, padx=3)
        self.lbl_y = Label(self, text="Y:")
        self.lbl_y.grid(row=6, column=0, sticky=W, padx=3)
        self.spin_y = Spinbox(self, width=8, from_=0, to=10000)
        self.spin_y.grid(row=6, column=1, sticky=W, padx=3)
        self.lbl_size = Label(self, text="Размер:")
        self.lbl_size.grid(row=5, column=4, sticky=W, padx=3)
        self.spin_size = Spinbox(self, width=8, from_=0, to=1000)
        self.spin_size.grid(row=5, column=5, sticky=W, padx=3)

        self.btn_inject = Button(self, text="Вставка", command=lambda: self.inject())
        self.btn_inject.grid(row=6, column=4, sticky=W+E)
        self.btn_save = Button(self, text="Сохранение", command=lambda: self.save())
        self.btn_save.grid(row=6, column=5, sticky=W+E)

    def centerWindow(self):
        w = 420
        h = 200
        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        x = (sw - w) / 2
        y = (sh - h) / 2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def load(self, action):
        if action == 'image':
            self.line_file.delete(0, END)
            path_to_file = filedialog.askopenfilename(initialdir="res/training_imgs")
            self.line_file.insert(0, path_to_file)
        elif action == 'model':
            self.line_model.delete(0, END)
            path_to_file = filedialog.askopenfilename(initialdir="res")
            self.line_model.insert(0, path_to_file)
            if path_to_file.find('context') == -1:
                self.btn_show_gen['state'] = NORMAL
                self.spin_size['state'] = NORMAL
            else:
                self.btn_show_gen['state'] = DISABLED
                self.spin_size.delete(0, END)
                self.spin_size.insert(0, '64')
                self.spin_size['state'] = DISABLED

    def show_graph(self, which):
        if which == 'original':
            path_to_file = self.line_file.get()
            image = PILimg.open(path_to_file)
            plt.figure()
            plt.axis("off")
            plt.title("Оригинал")
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            plt.show()
        if which == 'generated':
            path_to_file = self.line_model.get()
            generator.load_state_dict(torch.load(path_to_file))
            generator.eval()
            generator.to(device)

            fixed_noise = torch.randn(1, gen_input, 1, 1).to(device=device)
            global generated_samples
            generated_samples = generator(fixed_noise)
            generated_samples = generated_samples.cpu().detach()

            mean = 0.5
            std = 0.5
            unnorm = transforms.Normalize((-mean / std,), (1.0 / std,))
            generated_samples = unnorm(generated_samples)

            plt.figure()
            plt.axis("off")
            plt.title("Сгенерированный образец")
            plt.imshow(np.transpose(vutils.make_grid(generated_samples, normalize=False).cpu()))
            plt.show()

    def inject(self):
        global modified
        global generated_samples
        if self.line_model.get().find('context') == -1:
            path_to_file = self.line_file.get()
            original = PILimg.open(path_to_file)
            generated = transforms.ToPILImage()(generated_samples[0])
            x = int(self.spin_x.get())
            y = int(self.spin_y.get())
            size = int(self.spin_size.get())
            modified = original.copy()
            pixels_orig = modified.load()
            pix_orig_line = list()
            for px in range(x, x + size):
                pix_orig_line.append(pixels_orig[px, y])
            pix_orig_line = sorted(pix_orig_line)
            median_orig = pix_orig_line[int(len(pix_orig_line) / 2)]

            gen = generated.copy()
            pixels_gen = gen.load()
            pix_gen_line = list()
            for px in range(gen.size[0]):
                pix_gen_line.append(pixels_gen[px, 0])
            pix_gen_line = sorted(pix_gen_line)
            median_gen = pix_gen_line[int(len(pix_gen_line) / 2)]

            dif = median_gen - median_orig
            for px in range(gen.size[0]):
                for py in range(gen.size[1]):
                    pixels_gen[px, py] = pixels_gen[px, py] - dif

            modified.paste(gen.resize((size, size)), (x, y))
        else:
            path_to_file = self.line_file.get()
            original = PILimg.open(path_to_file)
            x = int(self.spin_x.get())
            y = int(self.spin_y.get())
            size = int(self.spin_size.get())
            tmp = 'res/context_out/masking.jpg'
            modified = original.copy()
            masking = modified.crop((x - int(size / 2), y - int(size / 2), x + 1.5 * size, y + 1.5 * size))
            masking.save(tmp)

            path_to_file = self.line_model.get()
            context_generator.load_state_dict(torch.load(path_to_file))
            context_generator.eval()
            context_generator.to(device)

            transforms_ = [
                transforms.Grayscale(1),
                # transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
            test_dataloader = DataLoader(
                ImageDataset(tmp, transforms_=transforms_, image_size=2*size, mask_size=size, mode="val"),
                batch_size=1,
                num_workers=1,
            )

            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            samples, masked_samples, i = next(iter(test_dataloader))
            masked_samples = Var(masked_samples.type(Tensor))
            i = i[0].item()

            gen_mask = context_generator(masked_samples)
            generated_samples = masked_samples.clone()
            generated_samples[:, :, i: i + size, i: i + size] = gen_mask
            generated_samples = generated_samples.cpu().detach()

            mean = 0.5
            std = 0.5
            unnorm = transforms.Normalize((-mean / std,), (1.0 / std,))
            generated_samples = unnorm(generated_samples)

            generated = transforms.ToPILImage()(generated_samples[0])
            modified.paste(generated, (x - int(size / 2), y - int(size / 2)))
            os.remove(tmp)

        plt.figure()
        plt.axis("off")
        plt.title("Модифицированное изображение")
        plt.imshow(modified, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def save(self):
        global modified
        path_to_save = filedialog.asksaveasfilename(defaultextension='jpg', filetypes=[("Image", '*.jpg;*.pgm;*.png')],
                                                    initialdir="res/out/results")
        if path_to_save == '':
            return
        modified.save(path_to_save)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_num > 0) else "cpu")
    generator = Generator(gen_input)
    context_generator = ContextGenerator(1)
    generated_samples = []
    modified = PILimg.Image
    window = Tk()
    app = App(window)
    window.mainloop()
