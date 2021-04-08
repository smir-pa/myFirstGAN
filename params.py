# В комментариях указаны предыдущие значения
# Путь к данным
path = "eyes/all"
# Число рабочих для загрузчика
workers = 2
# Объем обучающих данных
batch_size = 32  # 32
# Размер изображений
image_size = 64  # 64
# Число каналов изображений. (цвет: 3)
channels = 3
# Размер входного вектора генератора
gen_input = 100  # 100
# Размер карты признаков в генераторе/дискриминаторе
gen_fms = 64  # 64
dis_fms = 64  # 64
# Число эпох обучения
epochs = 25
# Степень обучения для оптимизаторов
l_rate = 0.0002
# Параметр для оптимизатора Adam (стохастический градиентный спуск)
beta1 = 0.5
# Количество доступных GPU (0 для использования CPU)
gpu_num = 1
