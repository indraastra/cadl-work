import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from cadl import datasets
from cadl import draw


A = 32
B = 32
C = 3
T = 25
n_enc = 256
n_z = 128
n_dec = 256
read_n = 9
write_n = 9
batch_size = 5
n_epochs = 1000


def VanGogh():
    img_dir = "vangogh"
    images = [np.array(ImageOps.fit(Image.open(os.path.join(img_dir, f)), (A, B), Image.ANTIALIAS))
              for f in os.listdir(img_dir)]
    Xs = np.array(images)
    Xs = Xs.reshape((Xs.shape[0], -1))
    print(Xs.shape)
    return datasets.Dataset(Xs, split=[.8, .1, .1])

draw.train_dataset(
    VanGogh(),
    A, 
    B,
    C,
    T,
    n_enc,
    n_z,
    n_dec,
    read_n,
    write_n,
    batch_size,
    n_epochs)

