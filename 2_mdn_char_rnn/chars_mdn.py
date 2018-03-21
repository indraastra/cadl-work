import string
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize as imresize

import train_charrnn


def load_mdn(ckpt_dir, sess):
  model = train_charrnn.build_model_mdn(
      batch_size=1,
      sequence_length=None,
      n_layers=3,
      n_cells=64,
      learning_rate=.001)
  saver = tf.train.Saver()
  ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
  if ckpt_path:
      saver.restore(sess, ckpt_path)
      print("CharRNN model restored.")
  return model


def predict_from_original(sess, model, font, size=100):
  actual = train_charrnn.render_chars(string.ascii_letters, [font], size)
  actual = actual.reshape(4, 13, size, size, 3)
  predicted_zs = []
  for c in string.ascii_letters[-1] + string.ascii_letters[:-1]:
    X = train_charrnn.encode([c], font, size)
    Y_pred = sess.run(
        model['Y_pred'],
        feed_dict={
            model['X']: train_charrnn.preprocess_zs(X),
            model['keep_prob']: 1.0
        })
    predicted_zs.append(train_charrnn.postprocess_zs(Y_pred))
  actual = actual.reshape(4, 13, size, size, 3)
  predicted = np.array([train_charrnn.decode(z) for z in predicted_zs])
  predicted = predicted.reshape(4, 13, size, size, 3)
  return actual, predicted


def predict_iteratively(sess, model, font, size=100):
  actual = train_charrnn.render_chars(string.ascii_letters, [font], size)
  actual = actual.reshape(4, 13, size, size, 3)
  predicted_zs = [train_charrnn.encode(['a'], font, size)]
  for _ in range(len(string.ascii_letters)):
    X = predicted_zs[-1]
    Y_pred = sess.run(
        model['Y_pred'],
        feed_dict={
            model['X']: train_charrnn.preprocess_zs(X),
            model['keep_prob']: 1.0
        })
    predicted_zs.append(train_charrnn.postprocess_zs(Y_pred))
  actual = actual.reshape(4, 13, size, size, 3)
  predicted = np.array([train_charrnn.decode(z) for z in predicted_zs])
  predicted[0, :] = predicted[-1, :]
  predicted = predicted[:-1].reshape(4, 13, size, size, 3)
  return actual, predicted


def show_comparison(actual, predicted):
  plt.subplot(2, 1, 1)
  plt.imshow(np.hstack(np.hstack(actual)), cmap='gray')
  plt.title('MDN predicts next character')
  plt.ylabel('actual')

  plt.subplot(2, 1, 2)
  plt.imshow(np.hstack(np.hstack(predicted)), cmap='gray')
  plt.ylabel('prediction')

  plt.show()
