import tensorflow as tf

from DATA_PRO import *
from PARAMETERS import Parameter as pm
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout,Dense
import numpy as np





class Cnn(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, num_classes):
        super(Cnn, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size,  embedding_size)
        self.conv_layer1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
        self.pool_layer1 = layers.GlobalMaxPool1D()
        self.conv_layer2 = layers.Conv1D(filters=32, kernel_size=4, activation='relu', padding='same')
        self.pool_layer2 = layers.GlobalMaxPool1D()
        self.conv_layer3 = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')
        self.pool_layer3 = layers.GlobalMaxPool1D()
        self.dense_layer = layers.Dense(units=64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x1 = self.conv_layer1(x)
        x1 = self.pool_layer1(x1)
        x2 = self.conv_layer2(x1)
        x2 = self.pool_layer2(x2)
        x3 = self.conv_layer3(x2)
        x3 = self.pool_layer3(x3)
        x3= self.dense_layer(x3)

        y = self.output_layer(x3)
        return y




