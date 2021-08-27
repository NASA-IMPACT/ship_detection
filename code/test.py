import tensorflow as tf
import numpy as np

def create_model():
    cnn_model = tf.keras.Sequential()
    cnn_model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(64))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.Dense(10))
    cnn_model.add(tf.keras.layers.Activation('softmax'))
    return cnn_model

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def pred(model, x):
    y_ = model(x)
    return tf.argmax(y_, axis=1)
