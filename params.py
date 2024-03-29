# Путь к данным
path = "res/training_imgs/normal_handmade"
# Число рабочих для загрузчика
workers = 4
# Объем обучающих данных
batch_size = 32
# Размер изображений
image_size = 64
# Число каналов изображений. (цвет: 3; оттенки серого: 1)
channels = 1
# Размер входного вектора генератора
gen_input = 100
# Размер карты признаков в генераторе/дискриминаторе
gen_fms = 64
dis_fms = 64
# Число эпох обучения
epochs = 500
# Степень обучения для оптимизаторов
dis_l_rate = 0.0002
gen_l_rate = 0.00001
# Параметр для оптимизатора Adam (стохастический градиентный спуск)
beta1 = 0.5
beta2 = 0.999
# Количество доступных GPU (0 для использования CPU)
gpu_num = 1
