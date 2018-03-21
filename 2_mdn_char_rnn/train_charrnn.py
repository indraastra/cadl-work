"""Character-level Recurrent Neural Network.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

######
### THIS HAS BEEN MODIFIED from the original version at:
###      pkmital/pycadl/master/cadl/charrnn.py
### The embedding baked into the charrnn model has been spliced out so that the
### pre-trained VAEGAN can directly provide embedding vectors to the modified
### char-rnn model. In other words, the VAEGAN provides the encoding/decoding
### capabilities while the RNN predicts embeddings from embeddings.
######

import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from scipy.special import logsumexp
import os
import random
import sys
import collections
import gzip
import string
from cadl import utils

######
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import chars_vaegan

sys.path.append("/home/vishal/Workspace/nn-ocr")

# From indraastra/nn-ocr.
from dataset import dataset, en
import fonts as font_utils

SIZE = 100

LABELS = en.get_labels()
FONTS = [font_utils.load_font(f) for f in en.get_fonts(shuffle=False)][:200]

tf.reset_default_graph()
sess = tf.InteractiveSession()
vaegan_model = chars_vaegan.load_vaegan("vaegan_model", sess)


def preprocess(img):
  img = imresize(img, (SIZE, SIZE))
  return img / 255.


def postprocess(recon):
  return (np.clip(recon / recon.max(), 0, 1) * 255).astype(np.uint8)


def encode(chars, font, size=SIZE):
  np_imgs = [np.asarray(font_utils.char_to_glyph(char, font, size), np.uint8)
             for char in chars]
  imgs_xs = [preprocess(np.tile(np.expand_dims(np_img, -1), (1, 1, 3)))
             for np_img in np_imgs]
  zs = sess.run(vaegan_model['z'], feed_dict={vaegan_model['x']: imgs_xs})
  return zs


def decode(zs):
  recon = sess.run(vaegan_model['x_tilde'],
                   feed_dict={vaegan_model['z']: zs.reshape(-1, 64)})
  return [postprocess(img) for img in recon]

def render_char(char, font, size=SIZE):
    np_img = np.asarray(font_utils.char_to_glyph(char, font, size), np.uint8)
    return np.tile(np.expand_dims(np_img, -1),
                   (1, 1, 3))

def render_chars(chars, fonts, size=SIZE):
    return np.array([[render_char(char, font, size) for char in chars]
                     for font in fonts])

def show_images(imgs):
    plt.ion()
    plt.imshow(np.hstack(np.hstack(imgs)), cmap='gray')
    plt.show()

def save_images(imgs, path):
    plt.imsave(arr=np.hstack(np.hstack(imgs)), fname=path, cmap='gray')

def gausspdf(x, mean, sigma):
    return tf.exp(-(x - mean)**2 / (2 * sigma**2)) / (tf.sqrt(2.0 * np.pi) * sigma)

def nll(x, means, sigmas, weights):
    p = np.exp(-(x - means)**2 / (2 * sigmas**2)) / (np.sqrt(2.0 * np.pi) * sigmas)
    weighted = p * weights
    sump = np.sum(weighted, axis=-1)
    negloglike = -np.log(np.maximum(sump, 1e-10))
    return negloglike

def nll_better(x, means, sigmas, weights):
    top = -(x - means)**2 / (2 * sigmas**2)
    bottom = np.log(weights) - np.log(np.sqrt(2.0 * np.pi)) - np.log(sigmas)
    negloglike = -logsumexp(top + bottom, axis=-1)
    return negloglike

def nll_tf(x, means, sigmas, weights):
    p = tf.exp(-(x - means)**2 / (2 * sigmas**2)) / (tf.sqrt(2.0 * np.pi) * sigmas)
    weighted = p * weights
    sump = tf.reduce_sum(weighted, axis=-1)
    negloglike = -tf.log(tf.maximum(sump, 1e-10))
    return negloglike

def nll_better_tf(x, means, sigmas, weights):
    top = -(x - means)**2 / (2 * sigmas**2)
    bottom = tf.log(weights) - tf.log(sigmas) - np.log(np.sqrt(2.0 * np.pi)) 
    negloglike = -tf.reduce_logsumexp(top + bottom, axis=-1)
    return negloglike

def preprocess_zs(zs):
  return zs / 10 + .5

def postprocess_zs(zs):
  return (zs - .5) * 10

#def nll(X, mean, sigma, weights):
#    return tf.reduce_logsumexp(-(x - mean)**2 / (2 * sigma**2)) -  (tf.sqrt(2.0 * np.pi) * sigma)

#####
def build_model_mdn(batch_size=1,
                sequence_length=1,
                n_layers=5,
                n_neurons=(128, 256, 1024, 1024, 256),
                n_cells=64,
                n_gaussians=64,
                gradient_clip=20.0,
                learning_rate=0.001):
    # Encoder and decoder are now provided externally.
    #encoder = collections.OrderedDict(zip(vocab, range(n_chars)))
    #decoder = collections.OrderedDict(zip(range(n_chars), vocab))

    X = tf.placeholder(tf.float32, [None, n_cells], name='X')
    Y = tf.placeholder(tf.float32, [None, n_cells], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    #with tf.variable_scope('embedding'):
    #    # Then slice each sequence element, giving us sequence number of
    #    # batch x 1 x n_chars Tensors
    #    Xs = tf.split(X, axis=1, num_or_size_splits=sequence_length)
    #    # Get rid of singleton sequence element dimension
    #    Xs = [tf.squeeze(X_i, [1]) for X_i in Xs]

    with tf.variable_scope('rnn'):
        #cells = tf.contrib.rnn.MultiRNNCell([
        #    tf.contrib.rnn.DropoutWrapper(
        #        tf.contrib.rnn.BasicLSTMCell(
        #            num_units=n_cells, forget_bias=0.0, state_is_tuple=True),
        #        output_keep_prob=keep_prob) for _ in range(n_layers)
        #])
        #initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
        ## returns a length sequence length list of outputs, one for each input
        #outputs, final_state = tf.contrib.rnn.static_rnn(
        #    cells, Xs, initial_state=initial_state)
        ## now concat the sequence length number of batch x n_cells Tensors to
        ## give [sequence_length x batch, n_cells]
        #outputs = tf.reshape(
        #    tf.concat(axis=1, values=outputs), [-1, sequence_length, n_cells])
        #outputs_flat = tf.reshape(
        #    tf.concat(axis=1, values=outputs), [-1, n_cells])
        pass

    with tf.variable_scope('mdn'):
        current_input = X
        for layer_i in range(len(n_neurons)):
            current_input = tf.layers.dense(
              inputs=current_input,
              units=n_neurons[layer_i],
              activation=tf.nn.relu, 
              name='layer/' + str(layer_i))
        #for layer_i in range(1, len(n_neurons)):
        #    current_input = tf.layers.conv1d(
        #      inputs=current_input,
        #      filters=n_neurons[layer_i],
        #      kernel_size=5,
        #      activation=tf.nn.relu)
              

        means = tf.reshape(
            tf.layers.dense(inputs=current_input,
                       units=n_cells * n_gaussians,
                       activation=tf.nn.relu,
                       name='means'), [-1, n_cells, n_gaussians])
        sigmas = tf.maximum(
            tf.reshape(
                tf.layers.dense(inputs=current_input,
                           units=n_cells * n_gaussians,
                           activation=tf.nn.relu,
                           name='sigmas'), [-1, n_cells, n_gaussians]), 1e-10)
        weights = tf.nn.softmax(tf.reshape(
            tf.layers.dense(inputs=current_input,
                       units=n_cells * n_gaussians,
                       activation=tf.nn.relu,
                       name='weights'), [-1, n_cells, n_gaussians]))
        #weights = tf.reshape(
        #    tfl.linear(inputs=current_input,
        #               num_outputs=n_cells * n_gaussians,
        #               activation_fn=tf.nn.softmax,
        #               scope='weights'), [-1, sequence_length, n_cells, n_gaussians])
        i0 = tf.multiply(means, weights)
        i1 = tf.reduce_sum(i0, axis=-1)
        Y_pred = tf.reshape(i1, [-1, n_cells])

    with tf.variable_scope('loss'):
        Y_3d = tf.reshape(Y, [-1, n_cells, 1])
        nll = nll_tf(Y_3d, means, sigmas, weights)
        cost = tf.reduce_mean(tf.reduce_mean(nll, 1))
        #cost = tf.losses.mean_squared_error(Y, Y_pred)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    model = {
        'X': X,
        'Y': Y,
        'Y_pred': Y_pred,
        'keep_prob': keep_prob,
        'cost': cost,
        'updates': optimizer,
        #'initial_state': initial_state,
        #'final_state': final_state,
    }
    return model


#def build_model(batch_size=1,
#                sequence_length=1,
#                n_layers=5,
#                n_neurons=(64, 256, 1024, 256, 64),
#                n_cells=64,
#                n_gaussians=16,
#                gradient_clip=20.0,
#                learning_rate=0.001):
#    # Encoder and decoder are now provided externally.
#    #encoder = collections.OrderedDict(zip(vocab, range(n_chars)))
#    #decoder = collections.OrderedDict(zip(range(n_chars), vocab))
#
#    X = tf.placeholder(tf.float32, [None, sequence_length, n_cells], name='X')
#    Y = tf.placeholder(tf.float32, [None, sequence_length, n_cells], name='Y')
#    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#
#    #with tf.variable_scope('embedding'):
#    #    # Then slice each sequence element, giving us sequence number of
#    #    # batch x 1 x n_chars Tensors
#    #    Xs = tf.split(X, axis=1, num_or_size_splits=sequence_length)
#    #    # Get rid of singleton sequence element dimension
#    #    Xs = [tf.squeeze(X_i, [1]) for X_i in Xs]
#
#    with tf.variable_scope('rnn'):
#        #cells = tf.contrib.rnn.MultiRNNCell([
#        #    tf.contrib.rnn.DropoutWrapper(
#        #        tf.contrib.rnn.BasicLSTMCell(
#        #            num_units=n_cells, forget_bias=0.0, state_is_tuple=True),
#        #        output_keep_prob=keep_prob) for _ in range(n_layers)
#        #])
#        #initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
#        ## returns a length sequence length list of outputs, one for each input
#        #outputs, final_state = tf.contrib.rnn.static_rnn(
#        #    cells, Xs, initial_state=initial_state)
#        ## now concat the sequence length number of batch x n_cells Tensors to
#        ## give [sequence_length x batch, n_cells]
#        #outputs = tf.reshape(
#        #    tf.concat(axis=1, values=outputs), [-1, sequence_length, n_cells])
#        #outputs_flat = tf.reshape(
#        #    tf.concat(axis=1, values=outputs), [-1, n_cells])
#        pass
#
#    with tf.variable_scope('mdn'):
#        current_input = X
#        for layer_i in range(1, len(n_neurons)):
#            current_input = tfl.linear(
#              inputs=current_input,
#              num_outputs=n_neurons[layer_i],
#              activation_fn=tf.nn.relu, 
#              scope='layer/' + str(layer_i))
#
#        means = tf.reshape(
#            tfl.linear(inputs=current_input,
#                       num_outputs=n_cells * n_gaussians,
#                       activation_fn=tf.nn.relu,
#                       scope='means'), [-1, sequence_length, n_cells, n_gaussians])
#        #means = tf.reshape(X, [-1, sequence_length, n_cells, n_gaussians])
#        sigmas = tf.maximum(
#            tf.reshape(
#                tfl.linear(inputs=current_input,
#                           num_outputs=n_cells * n_gaussians,
#                           activation_fn=tf.nn.sigmoid,
#                           scope='sigmas'), [-1, sequence_length, n_cells, n_gaussians]), 1e-10)
#        weights = tf.nn.softmax(tf.reshape(
#            tfl.linear(inputs=current_input,
#                       num_outputs=n_cells * n_gaussians,
#                       activation_fn=tf.nn.sigmoid,
#                       scope='weights'), [-1, sequence_length, n_cells, n_gaussians]))
#        i0 = tf.multiply(means, weights)
#        i1 = tf.reduce_sum(i0, axis=3)
#
#        def _debug_print_func(t):
#            print('tensor shape: {}'.format(t.shape))
#            return False
#        debug_print_op = tf.py_func(_debug_print_func, [i1], [tf.bool])
#        with tf.control_dependencies(debug_print_op):
#          Y_pred = tf.reshape(i1, [-1, sequence_length, n_cells])
#
#    with tf.variable_scope('loss'):
#        #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [
#        #    tf.reshape(tf.concat(axis=1, values=Y), [-1])
#        #], [tf.ones([batch_size * sequence_length])])
#        #loss = tf.losses.mean_squared_error(Y, Y_pred)
#        Y_3d = tf.reshape(Y, [-1, sequence_length, n_cells, 1])
#        p = gausspdf(Y_3d, means, sigmas)
#        weighted = weights * p
#        sump = tf.reduce_sum(weighted, 2)
#        negloglike = -tf.log(tf.maximum(sump, 1e-10))
#        cost = tf.reduce_mean(tf.reduce_mean(negloglike, 1)) / batch_size
#
#    with tf.name_scope('optimizer'):
#        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#        gradients = []
#        clip = tf.constant(gradient_clip, name="clip")
#        for grad, var in optimizer.compute_gradients(cost):
#            gradients.append((tf.clip_by_value(grad, -clip, clip), var))
#        updates = optimizer.apply_gradients(gradients)
#
#    model = {
#        'X': X,
#        'Y': Y,
#        'Y_pred': Y_pred,
#        'keep_prob': keep_prob,
#        'cost': cost,
#        'updates': updates,
#        #'initial_state': initial_state,
#        #'final_state': final_state,
#    }
#    return model


def train(txt,
          batch_size=128,
          sequence_length=52,
          n_cells=64,
          n_layers=3,
          learning_rate=1e-4,
          max_iter=50000,
          ckpt_name="model.ckpt",
          keep_prob=1.0):
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model_mdn(
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_layers=n_layers,
            n_cells=n_cells,
            learning_rate=learning_rate)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess.run(init_op)
        ckpt_path = tf.train.latest_checkpoint(os.path.dirname(ckpt_name))
        if ckpt_path:
            saver.restore(sess, ckpt_path)
            print("CharRNN model restored.")

        print("Beginnning training!")
        cursor = 0
        it_i = 0
        print_step = 500
        save_step = 1000
        avg_cost = 0
        Xs_full, Ys_full = [], []
        for font in FONTS:
            try:
              Xs_full.extend(encode(txt[:sequence_length], font))
              Ys_full.extend(encode(txt[1:sequence_length + 1], font))
              #Ys_full.append(encode(txt[:sequence_length], font))
            except TypeError:
              FONTS.remove(font)
              continue
        Xs_full = np.array(Xs_full)
        Ys_full = np.array(Ys_full)
        while it_i < max_iter:
            #Xs, Ys = [], []
            #while len(Xs) < batch_size:
            #    font = random.choice(FONTS)
            #    try:
            #      Xs = encode(txt[cursor:cursor + sequence_length], font)
            #      Ys = encode(txt[cursor + 1:cursor + sequence_length + 1], font)
            #    except TypeError:
            #      FONTS.remove(font)
            #      continue
            #    cursor += sequence_length
            #    if (cursor + 1) >= len(txt) - sequence_length - 1:
            #        cursor = np.random.randint(0, high=sequence_length)
            #Xs = np.array(Xs) / 10 + .5
            #Ys = np.array(Ys) / 10 + .5
            idxs = np.random.choice(len(Xs_full), batch_size)
            Xs = Xs_full[idxs]
            Ys = Ys_full[idxs]
            Xs = preprocess_zs(Xs)
            Ys = preprocess_zs(Ys)
            feed_dict = {
                model['X']: Xs,
                model['Y']: Ys,
                model['keep_prob']: keep_prob
            }
            out = sess.run(
                [model['cost'], model['updates']], feed_dict=feed_dict)
            avg_cost += out[0]

            if (it_i + 1) % print_step == 0:
                print(Xs[-1].shape)
                Ys_pred = sess.run(
                    model['Y_pred'],
                    feed_dict={
                        model['X']: Xs,
                        model['keep_prob']: 1.0
                    })
                if isinstance(txt[0], str):
                    # Save original and predictions
                    print('Xs min/max: ', np.min(Xs), np.max(Xs))
                    print('Yx min/max: ', np.min(Ys), np.max(Ys))
                    print('Ys min/max: ', np.min(Ys), np.max(Ys))
                    print('Ys_pred min/max: ', np.min(Ys_pred), np.max(Ys_pred))
                    print(Ys[0])
                    print(Ys[1])
                    print(Ys_pred[0])
                    print(Ys_pred[1])
                    Xs_orig = postprocess_zs(Xs)
                    Ys_pred = postprocess_zs(Ys_pred)
                    print('Ys_pred min/max: ', np.min(Ys_pred), np.max(Ys_pred))
                    imgs = [decode(Xs_orig), decode(Ys_pred)]
                    save_images(imgs, 'imgs/combo_{:06}.png'.format(it_i))

                print(it_i, avg_cost / print_step)
                avg_cost = 0

            if (it_i + 1) % save_step == 0:
                save_path = saver.save(sess, ckpt_name, global_step=it_i)
                print("Model saved in file: %s" % save_path)

            print(it_i, out[0])
            it_i += 1

        return model


def test_trump(max_iter=50000):
    """Summary
    Parameters
    ----------
    max_iter : int, optional
        Description
    """
    utils.download('https://s3.amazonaws.com/cadl/models/trump.txt')
    #with open('trump.txt', 'r') as fp:
    #    txt = fp.read()
    txt = string.ascii_letters + 'a'
    #train(txt, ckpt_name='./mdn_model/trump/trump.ckpt', max_iter=max_iter, batch_size=64)
    train(txt, ckpt_name='./mdn_model/alpha/alpha.ckpt', max_iter=max_iter, batch_size=64)
    #train('xy' * 10000, ckpt_name='./mdn_model/test/test.ckpt', max_iter=max_iter, batch_size=32)


if __name__ == '__main__':
  #imgs = render_chars('foo'*50, random.sample(FONTS, 5))
  #show_images(imgs)
  #save_images(imgs, 'test.png')
  #_ = input('press enter to continue...')
  test_trump()

