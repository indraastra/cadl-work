import os
import sys

from flask import Flask, json, render_template
import numpy as np
import tensorflow as tf

from embedding import word2vec_optimized as word2vec


FLAGS = word2vec.FLAGS

if not FLAGS.save_path:
  print("--save_path must be specified.")
  sys.exit(1)
if not os.path.exists(FLAGS.save_path):
  print("--save_path does not exist:", FLAGS.save_path)
  sys.exit(1)
opts = word2vec.Options()
session = tf.Session()
model = word2vec.Word2Vec(opts, session)
model.saver.restore(session, tf.train.latest_checkpoint(opts.save_path))

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('tsne.html')


@app.route('/analogy')
def analogy():
  return render_template('analogy.html')


@app.route('/analogy/<word_a>/<word_b>/<word_c>')
def get_analogy(word_a, word_b, word_c):
  return json.jsonify({
    'word_a': word_a,
    'word_b': word_b,
    'word_c': word_c,
    'word_d': model.analogy(bytes(word_a, 'utf-8'),
                            bytes(word_b, 'utf-8'),
                            bytes(word_c, 'utf-8')).decode('utf-8')
  })


@app.route('/tsne')
def tsne():
  return render_template('tsne.html')


@app.route('/embedding/<word>')
def get_embedding(word):
  emb = model.get_embedding(bytes(word, 'utf-8'))
  return json.jsonify({
    'word': word,
    'vec': list(emb)
  })


@app.route('/nearby/<word>')
def get_nearby(word):
  nearby = model.get_nearby(bytes(word, 'utf-8'))
  return json.jsonify({
    'word': word,
    'nearby': nearby
  })


if __name__ == '__main__':
  """Train a word2vec model."""
  app.run(debug=False, host='0.0.0.0')
