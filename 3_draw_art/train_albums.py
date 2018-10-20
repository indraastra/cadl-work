import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from cadl import datasets
from cadl import draw


A = 64
B = 64
C = 3
T = 20
n_enc = 512
n_z = 196
n_dec = 512
read_n = 15
write_n = 15
batch_size = 64
n_epochs = 500
img_dir = "albums_data"


images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
print("Found", len(images), "album covers")
draw.train_input_pipeline(
    [os.path.join(img_dir, f) for f in os.listdir(img_dir)],
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

