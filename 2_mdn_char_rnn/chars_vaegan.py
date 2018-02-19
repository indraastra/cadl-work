import numpy as np
import tensorflow as tf
from skimage.transform import resize as imresize

from cadl import vaegan


def load_vaegan(ckpt_name, sess):
  n_epochs=100
  filter_sizes=[3, 3, 3, 3]
  n_filters=[100, 100, 100, 100]
  crop_shape=[100, 100, 3]
  model = vaegan.VAEGAN(
        input_shape=[None] + crop_shape,
        convolutional=True,
        variational=True,
        n_filters=n_filters,
        n_hidden=None,
        n_code=64,
        filter_sizes=filter_sizes,
        activation=tf.nn.elu)
  saver = tf.train.Saver()
  ckpt_path = tf.train.latest_checkpoint(ckpt_name)
  if ckpt_path:
    saver.restore(sess, ckpt_path)
    print("VAE model restored.")
  return model


def preprocess(img, crop_factor=0.8):
  crop = np.min(img.shape[:2])
  r = (img.shape[0] - crop) // 2
  c = (img.shape[1] - crop) // 2
  cropped = img[r:r + crop, c:c + crop]
  r, c, *d = cropped.shape
  if crop_factor < 1.0:
    amt = (1 - crop_factor) / 2
    h, w = int(c * amt), int(r * amt)
    cropped = cropped[h:-h, w:-w]
  rsz = imresize(cropped, (100, 100))
  return rsz
