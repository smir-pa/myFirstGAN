
import os.path
import sys
import urllib.request
import urllib.error
from datetime import datetime


def data_load(file):
    if not os.path.exists(file):
        print("There's no such file")
        sys.exit()
    else:
        d = list()
        with open(file) as f:
            for line in f:
                d.append(line.strip().split(' '))
        return d


def download(lbl, key):
    url = 'http://peipa.essex.ac.uk/pix/mias/'
    if not os.path.exists("res/mias/" + key):
        os.mkdir("res/mias/" + key)
    for img in lbl[key]:
        try:
            urllib.request.urlretrieve(url + img + ".pgm", "res/mias/" + key + "/" + img + ".pgm")
        except urllib.error.HTTPError:
            print("No such file", img)


diagnosis = {'CALC': [],
             'CIRC': [],
             'SPIC': [],
             'MISC': [],
             'ARCH': [],
             'ASYM': [],
             'NORM': []}
counts = dict()
t0 = datetime.now()
data = data_load("res/Info.txt")
print("Превью:\n", data[:3])
for el in data:
    diagnosis[el[2]].append(el[0])
for k, v in diagnosis.items():
    counts[k] = len(v)
print("Количество каждого:\n", counts)
for k in diagnosis.keys():
    download(diagnosis, k)
print("Времени прошло, с:", (datetime.now() - t0).seconds)
