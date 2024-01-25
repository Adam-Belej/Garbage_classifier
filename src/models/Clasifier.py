import typing
import tensorflow as tf
import matplotlib.pyplot as plt
import utils.network_utils as nu


class Classifier:

    def __init__(self,
                 num_of_classes: int,
                 epochs: int = 5,
                 batchsize: int = 12,
                 valsplit: int = 0.001):
        self.num_of_classes = num_of_classes
        self.epochs = epochs
        self.batchsize = batchsize
        self.valsplit = valsplit


    def load_pretrained_model(self, path: str):
        self.model = tf.keras.models.load_model(filepath=path)


    def export_model(self, path: str):
        tf.keras.Model.save(self.model, filepath=path)

    def train(self, training_data, validation_data):
        history = self.model.fit(training_data,
                                 epochs=self.epochs,
                                 validation_data=validation_data,
                                 steps_per_epoch=100,
                                 shuffle=True)
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


    def make_graph_from_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        print("Calculating the loss")
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)
        print("The results are being visualized")
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Tréninková přesnost')
        plt.plot(epochs_range, val_acc, label='Validační přesnost')
        plt.legend(loc='lower right')
        plt.title('Tréninková a validační přesnost')
        plt.axis('square')
        plt.subplot(1, 2, 2)

        plt.plot(epochs_range, loss, label='Tréninková ztráta')
        plt.plot(epochs_range, val_loss, label='Validační ztráta')
        plt.legend(loc='upper right')
        plt.title('Tréninková a validační ztráta')
        plt.axis('square')
        plt.show()

    def test_accuracy(self, test_data_dir: str,
                      img_width: int = 512,
                      img_height: int = 512):
        test_ds = nu.load_dataset(data_dir=test_data_dir,
                        img_width=img_width,
                        img_height=img_height)

        self.evaluation = self.model.evaluate(test_ds)

