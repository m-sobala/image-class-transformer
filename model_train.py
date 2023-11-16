import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model

# Load in the cifar100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

image_size = 72

AugmentData = Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name = 'AugmentData'
)
AugmentData.layers[0].adapt(x_train)

