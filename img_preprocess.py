
import glob
import os
import pandas as pd
from PIL import Image


def convert_to_jpg(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print('Директории', out_dir, 'не существует. Создана директория', out_dir)
        os.mkdir(out_dir)
    if not os.path.exists(in_dir):
        print('Директории', in_dir, 'не существует')
        return -1
    count = 0
    for files in glob.glob(in_dir + '/*'):
        filepath, filename = os.path.split(files)
        out_file = filename[0:len(filename)-4] + '.jpg'
        im = Image.open(files)
        im.save(os.path.join(out_dir, out_file))
        count = count + 1
    print('Успешно сконвертировано', count, 'файлов из', in_dir, 'в', out_dir)


def crop_img(coords, in_dir, out_dir):
    if not os.path.exists(out_dir):
        print('Директории', out_dir, 'не существует. Создана директория', out_dir)
        os.mkdir(out_dir)
        os.mkdir(out_dir + '/1')
    if not os.path.exists(in_dir):
        print('Директории', in_dir, 'не существует')
        return -1
    data = pd.read_csv(coords)
    for index, row in data.iterrows():
        img = Image.open(in_dir + '/' + row['name'])
        cropped = img.crop((row['x'] - int(row['size']/2), row['y'] - int(row['size']/2),
                            row['x'] + 1.5*row['size'], row['y'] + 1.5*row['size']))
        if not os.path.exists(out_dir + '/' + row['name']):
            cropped.save(out_dir + '/1/' + row['name'])
        else:
            cropped.save(out_dir + '/1/' + row['name'][0:len(row['name'])-4] + '_1.jpg')


if __name__ == '__main__':
    convert_to_jpg('res/training_imgs/normal_pgm', 'res/training_imgs/normal_jpg')
    convert_to_jpg('res/training_imgs/nodules_pgm', 'res/training_imgs/nodules_jpg')

    crop_img('res/training_imgs/nodules_coords.csv',
             'res/training_imgs/nodules_jpg',
             'res/training_imgs/nodules_for_masking')
