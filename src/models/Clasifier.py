import typing
import numpy as np
import tensorflow as tf


class Classifier:

    def __init__(self,
                 epochs: int = 100,
                 batchsize: int = 32,
                 valsplit: int = 0.2):
        self.epochs = epochs
        self.batchsize = batchsize
        self.valsplit = valsplit


    def load_pretrained_model(self, path: str):
        self.model = tf.saved_model.load(path)


    def export_model(self, path: str):
        tf.saved_model.save(self.model, export_dir=path)


    def train(self, data):
        self.model.fit(data,
                       epochs=self.epochs,
                       batch_size=self.batchsize,
                       validation_split=self.valsplit)


    def classify(self, data: str, height: int, width: int, class_names: tuple):
        data_path = tf.keras.utils.get_file(origin=data)

        img = tf.keras.utils.load_img(
            data_path, target_size=(height, width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * max(score))
        )

