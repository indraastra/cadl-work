"""Interactive exploration of a word2vec model using analogies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import tensorflow as tf
import word2vec_optimized as word2vec


FLAGS = word2vec.FLAGS


def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
  """Train a word2vec model."""
  if not FLAGS.save_path:
    print("--save_path must be specified.")
    sys.exit(1)
  if not os.path.exists(FLAGS.save_path):
    print("--save_path does not exist:", FLAGS.save_path)
    sys.exit(1)
  opts = word2vec.Options()
  with tf.Graph().as_default(), tf.Session() as session:
    model = word2vec.Word2Vec(opts, session)
    # Perform a final save.
    model.saver.restore(session, tf.train.latest_checkpoint(opts.save_path))
    # E.g.,
    # [0]: model.analogy(b'france', b'paris', b'russia')
    # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
    word2vec._start_shell(locals())


if __name__ == "__main__":
  tf.app.run()
