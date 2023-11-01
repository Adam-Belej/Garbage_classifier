import tensorflow as tf
from Clasifier import Classifier


class Alfonzo(Classifier):

    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=4)
            ])
