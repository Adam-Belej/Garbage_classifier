import typing
import tensorflow as tf


class Classifier:

    def __init__(self,
                 num_of_classes: int,
                 epochs: int = 10,
                 batchsize: int = 32,
                 valsplit: int = 0.2):
        self.num_of_classes = num_of_classes
        self.epochs = epochs
        self.batchsize = batchsize
        self.valsplit = valsplit


    def load_pretrained_model(self, path: str):
        self.model = tf.saved_model.load(path)


    def export_model(self, path: str):
        tf.saved_model.save(self.model, export_dir=path)

    def train(self, training_data, validation_data):
        history = self.model.fit(training_data,
                                 epochs=self.epochs,
                                 batch_size=self.batchsize,
                                 validation_data=validation_data)
        return history


    def classify(self, data: str, height: int, width: int):
        img = tf.keras.utils.load_img(
            data, target_size=(height, width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions)

        return score
