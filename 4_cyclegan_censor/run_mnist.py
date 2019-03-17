import os
import numpy as np
import tensorflow as tf
from cadl import cycle_gan
from cadl import utils
from cadl.datasets import MNIST
from train_mnist import *


def preprocess(imgs):
    return imgs/127.5 - 1

def postprocess(imgs):
    return [(img + 1) / 2 for img in imgs]

if __name__ == '__main__':
    mnist = MNIST()
    mnist = np.repeat(mnist.train.images.reshape((-1, 28, 28, 1)), 3, axis=-1)[10000:10016]
    inputs = [pil2np(add_watermark_v1(np2pil(img), color=(0,255,0))) for img in mnist]
    utils.montage(postprocess(inputs), 'mnist_input.png')

    with tf.Graph().as_default(), tf.Session() as sess:
        net = cycle_gan.cycle_gan(img_size=28)
        saver = tf.train.Saver()
        ckpt_path = tf.train.latest_checkpoint('mnist')
        saver.restore(sess, ckpt_path)
        outputs = []
        for X in inputs:
            Y_fake = sess.run(net['Y_fake'], feed_dict={ net['X_real']: [X] })
            outputs.append(Y_fake[0])
        utils.montage(postprocess(outputs), 'mnist_output.png')
