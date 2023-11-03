import typing
import tensorflow as tf

class Classifier:


    def __init__(self, model_path: str = None):
        pass


    def load_pretrained_model(self, path: str):
        self.model = tf.saved_model.load(path)


    def export_model(self, path: str):
        tf.saved_model.save(self.model, export_dir=path)

    def train(self, data):
        pass


    def classify(self, data):
        pass


