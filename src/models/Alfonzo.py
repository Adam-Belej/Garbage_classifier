import tensorflow as tf
from models.Clasifier import Classifier


class Alfonzo(Classifier):

    def __init__(self,
                 num_of_classes: int):
        super().__init__(num_of_classes)
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=num_of_classes)
            ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics="accuracy")
