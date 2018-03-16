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
import numpy as np
import os
import random
import sys
import collections
import gzip
from cadl import utils

######
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import chars_vaegan

sys.path.append("/Users/vtalwar/Workspace/nn-ocr")

# From indraastra/nn-ocr.
from dataset import dataset, en
import fonts as font_utils

SIZE = 100

LABELS = en.get_labels()
FONTS = random.sample([font_utils.load_font(f) for f in en.get_fonts()], 200)

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

#####

def build_model(batch_size=1,
                sequence_length=1,
                n_layers=2,
                n_cells=64,
                gradient_clip=10.0,
                learning_rate=0.001):
    # Encoder and decoder are now provided externally.
    #encoder = collections.OrderedDict(zip(vocab, range(n_chars)))
    #decoder = collections.OrderedDict(zip(range(n_chars), vocab))

    X = tf.placeholder(tf.float32, [None, sequence_length, n_cells], name='X')
    Y = tf.placeholder(tf.float32, [None, sequence_length, n_cells], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.variable_scope('embedding'):
        # Then slice each sequence element, giving us sequence number of
        # batch x 1 x n_chars Tensors
        Xs = tf.split(X, axis=1, num_or_size_splits=sequence_length)
        # Get rid of singleton sequence element dimension
        Xs = [tf.squeeze(X_i, [1]) for X_i in Xs]

    with tf.variable_scope('rnn'):
        cells = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(
                    num_units=n_cells, forget_bias=0.0, state_is_tuple=True),
                output_keep_prob=keep_prob) for _ in range(n_layers)
        ])
        initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
        # returns a length sequence length list of outputs, one for each input
        outputs, final_state = tf.contrib.rnn.static_rnn(
            cells, Xs, initial_state=initial_state)
        # now concat the sequence length number of batch x n_cells Tensors to
        # give [sequence_length x batch, n_cells]
        outputs_flat = tf.reshape(
            tf.concat(axis=1, values=outputs), [-1, n_cells])

    with tf.variable_scope('prediction'):
        Y_pred = tf.reshape(tf.nn.sigmoid(outputs_flat), [-1, sequence_length, n_cells])

    with tf.variable_scope('loss'):
        #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [
        #    tf.reshape(tf.concat(axis=1, values=Y), [-1])
        #], [tf.ones([batch_size * sequence_length])])
        loss = tf.losses.mean_squared_error(Y, Y_pred)
        cost = tf.reduce_sum(loss) / batch_size

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = []
        clip = tf.constant(gradient_clip, name="clip")
        for grad, var in optimizer.compute_gradients(cost):
            gradients.append((tf.clip_by_value(grad, -clip, clip), var))
        updates = optimizer.apply_gradients(gradients)

    model = {
        'X': X,
        'Y': Y,
        'Y_pred': Y_pred,
        'keep_prob': keep_prob,
        'cost': cost,
        'updates': updates,
        'initial_state': initial_state,
        'final_state': final_state,
    }
    return model


def train(txt,
          batch_size=100,
          sequence_length=150,
          n_cells=64,
          n_layers=3,
          learning_rate=0.00001,
          max_iter=50000,
          gradient_clip=5.0,
          ckpt_name="model.ckpt",
          keep_prob=1.0):
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_layers=n_layers,
            n_cells=n_cells,
            gradient_clip=gradient_clip,
            learning_rate=learning_rate)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess.run(init_op)
        if os.path.exists(ckpt_name + '.index') or os.path.exists(ckpt_name):
            saver.restore(sess, ckpt_name)
            print("Model restored.")

        cursor = 0
        it_i = 0
        print_step = 10
        avg_cost = 0
        while it_i < max_iter:
            Xs, Ys = [], []
            font = random.choice(FONTS)
            for batch_i in range(batch_size):
                Xs.append(encode(txt[cursor:cursor + sequence_length], font))
                Ys.append(encode(txt[cursor + 1:cursor + sequence_length + 1], font))
                cursor += sequence_length
                if (cursor + 1) >= len(txt) - sequence_length - 1:
                    cursor = np.random.randint(0, high=sequence_length)

            feed_dict = {
                model['X']: Xs,
                model['Y']: Ys,
                model['keep_prob']: keep_prob
            }
            out = sess.run(
                [model['cost'], model['updates']], feed_dict=feed_dict)
            avg_cost += out[0]

            if (it_i + 1) % print_step == 0:
                Ys_pred = sess.run(
                    model['Y'],
                    feed_dict={
                        model['X']: np.array(Xs[-1])[np.newaxis],
                        model['keep_prob']: 1.0
                    })
                if isinstance(txt[0], str):
                    # Save original
                    imgs = decode(Xs[-1])
                    save_images(imgs, 'imgs/original_{}.png'.format(it_i))

                    # Save prediction
                    imgs = decode(Ys_pred[-1])
                    save_images(imgs, 'imgs/prediction_{}.png'.format(it_i))

                print(it_i, avg_cost / print_step)
                avg_cost = 0

                save_path = saver.save(sess, ckpt_name, global_step=it_i)
                print("Model saved in file: %s" % save_path)

            print(it_i, out[0], end='\r')
            it_i += 1

        return model


def infer(txt,
          ckpt_name,
          n_iterations,
          n_cells=200,
          n_layers=3,
          learning_rate=0.001,
          max_iter=5000,
          gradient_clip=10.0,
          init_value=[0],
          keep_prob=1.0,
          sampling='prob',
          temperature=1.0):
    """infer

    Parameters
    ----------
    txt : TYPE
        Description
    ckpt_name : TYPE
        Description
    n_iterations : TYPE
        Description
    n_cells : int, optional
        Description
    n_layers : int, optional
        Description
    learning_rate : float, optional
        Description
    max_iter : int, optional
        Description
    gradient_clip : float, optional
        Description
    init_value : list, optional
        Description
    keep_prob : float, optional
        Description
    sampling : str, optional
        Description
    temperature : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        sequence_length = len(init_value)
        model = build_model(
            txt=txt,
            batch_size=1,
            sequence_length=sequence_length,
            n_layers=n_layers,
            n_cells=n_cells,
            gradient_clip=gradient_clip,
            learning_rate=learning_rate)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        sess.run(init_op)
        saver.restore(sess, ckpt_name)
        print("Model restored.")

        state = []
        synth = [init_value]
        for s_i in model['final_state']:
            state += sess.run(
                [s_i.c, s_i.h],
                feed_dict={
                    model['X']: [synth[-1]],
                    model['keep_prob']: keep_prob
                })

        for i in range(n_iterations):
            # print('iteration: {}/{}'.format(i, n_iterations), end='\r')
            feed_dict = {model['X']: [synth[-1]], model['keep_prob']: keep_prob}
            state_updates = []
            for state_i in range(n_layers):
                feed_dict[model['initial_state'][state_i].c] = \
                    state[state_i * 2]
                feed_dict[model['initial_state'][state_i].h] = state[state_i * 2
                                                                     + 1]
                state_updates.append(model['final_state'][state_i].c)
                state_updates.append(model['final_state'][state_i].h)
            p = sess.run(model['probs'], feed_dict=feed_dict)[0]
            if sampling == 'max':
                p = np.argmax(p)
            else:
                p = p.astype(np.float64)
                p = np.log(p) / temperature
                p = np.exp(p) / np.sum(np.exp(p))
                p = np.random.multinomial(1, p.ravel())
                p = np.argmax(p)
            # Get the current state
            state = [
                sess.run(s_i, feed_dict=feed_dict) for s_i in state_updates
            ]
            synth.append([p])
            print(model['decoder'][p], end='')
            sys.stdout.flush()
            if model['decoder'][p] in ['.', '?', '!']:
                print('\n')
        print(np.concatenate(synth).shape)
    print("".join([model['decoder'][ch] for ch in np.concatenate(synth)]))
    return [model['decoder'][ch] for ch in np.concatenate(synth)]


if __name__ == '__main__':
  train('abcdef foo bar baz ' * 5000, ckpt_name='charrnn.ckpt')
  #imgs = render_chars('foo'*50, random.sample(FONTS, 5))
  #show_images(imgs)
  #save_images(imgs, 'test.png')
  _ = input('press enter to continue...')

