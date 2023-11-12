import typing
import tensorflow as tf
import matplotlib.pyplot as plt


class Classifier:

    def __init__(self,
                 num_of_classes: int,
                 epochs: int = 2,
                 batchsize: int = 16,
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
                                   validation_data=validation_data)
        self.history = history


    def classify(self, data: str, height: int, width: int):
        img = tf.keras.utils.load_img(
            data, target_size=(height, width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions)

        return score


    def make_graph_from_history(self, graph_path):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        print("Calculating the loss")
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)
        print("The results are being visualized")
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)

        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        plt.savefig(graph_path)