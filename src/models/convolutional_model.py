import tensorflow as tf
from models.Clasifier import Classifier


class Convolutional(Classifier):

    def __init__(self,
                 num_of_classes: int):
        super().__init__(num_of_classes)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_of_classes)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics="accuracy")
