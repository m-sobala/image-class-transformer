'''Tensorflow import'''
import tensorflow as tf

# Load in the cifar100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
