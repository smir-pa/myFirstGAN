# Путь к данным
path = "res/training_imgs/nodules_for_masking"
# Загрузка предобученной модели
load_pretrained_models = False
# Путь к модели генератора
gen_model_path = 'res/context_out/generator_context(05.05.2021 00-29-35).pt'
# Путь к модели дискриминатора
dis_model_path = 'res/context_out/discriminator_context(05.05.2021 00-29-35).pt'
# Число рабочих для загрузчика
workers = 4
# Объем обучающих данных
batch_size = 32
# Размер изображений
image_size = 128
# Размер маски
mask_size = int(image_size / 2)
# Интервал между выборками
sample_interval = 500
# Число каналов изображений. (цвет: 3; оттенки серого: 1)
channels = 1
# Размер входного вектора шума генератора
gen_input = 100
# Размер карты признаков в генераторе/дискриминаторе
gen_fms = 64
dis_fms = 64
# Число эпох обучения
epochs = 300
# Степень обучения для оптимизаторов
dis_l_rate = 0.00008
gen_l_rate = 0.00008
# Параметр для оптимизатора Adam (стохастический градиентный спуск)
beta1 = 0.5
beta2 = 0.999
# Количество доступных GPU (0 для использования CPU)
gpu_num = 1
# Расчет выходных измерений изображения дискриминатора
patch_h, patch_w = int(mask_size / 2 ** 3), int(mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)
