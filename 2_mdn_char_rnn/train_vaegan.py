import glob

from cadl import vaegan
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_images', 'data/en/**.png',
  'Glob pattern matching the input images.')
flags.DEFINE_string('output_ckpt', 'models/saved/vaegan',
  'Directory to store output checkpoints.')


def main(argv=None):
  dataset_files = glob.glob(FLAGS.input_images, recursive=True)
  print(len(dataset_files), 'files matched!')
  n_epochs=100
  filter_sizes=[3, 3, 3, 3]
  n_filters=[100, 100, 100, 100]
  crop_shape=[100, 100, 3]
  vaegan.train_vaegan(
      files=dataset_files,
      batch_size=64,
      n_epochs=n_epochs,
      crop_shape=crop_shape,
      crop_factor=0.8,
      input_shape=[72, 72, 3],
      convolutional=True,
      variational=True,
      n_filters=n_filters,
      n_hidden=None,
      n_code=64,
      filter_sizes=filter_sizes,
      activation=tf.nn.elu,
      ckpt_name=FLAGS.output_ckpt)


if __name__ == '__main__':
  tf.app.run()
