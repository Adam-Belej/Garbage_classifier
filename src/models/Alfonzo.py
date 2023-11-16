import tensorflow as tf
from models.Clasifier import Classifier


class Alfonzo(Classifier):

    def __init__(self,
                 num_of_classes: int):
        super().__init__(num_of_classes)
        self.model = tf.keras.Sequential([
                        tf.keras.layers.Rescaling(1./255),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(num_of_classes)
                        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics="accuracy")
