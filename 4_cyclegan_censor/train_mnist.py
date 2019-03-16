import random

import numpy as np
from PIL import Image, ImageDraw

from cadl import cycle_gan
from cadl.datasets import MNIST

def draw_random_rectangle(draw, img, color=(255, 255, 255)):
    x0 = random.randint(0, img.size[0])
    x1 = random.randint(x0, img.size[0])
    y0 = random.randint(0, img.size[1])
    y1 = random.randint(y0, img.size[1])
    alpha = random.randint(200, 255)
    draw.rectangle((x0, x1, y0, y1), fill=color + (alpha,))

def draw_x(draw, img, color):
    draw.line((0, 0) + img.size, fill=color, width=10)
    draw.line((0, img.size[1], img.size[0], 0), fill=color, width=10)
    
def add_watermark_v1(img):
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    draw_random_rectangle(draw, img)
    return Image.alpha_composite(img.convert('RGBA'), overlay)

def np2pil(np_img):
    return Image.fromarray((np_img * 255).astype(np.uint8))

def pil2np(pil_img):
    return np.array(pil_img.convert('RGB')).astype(np.float) / 255

mnist = MNIST()

imgs_in = np.repeat(mnist.train.images.reshape((-1, 28, 28, 1)), 3, axis=-1)[:1000]
print(imgs_in.shape)
imgs_out = np.array([pil2np(add_watermark_v1(np2pil(img))) for img in imgs_in])
print(imgs_out.shape)

cycle_gan.train(imgs_in, imgs_out, ckpt_path='mnist', img_size=28, n_epochs=1)