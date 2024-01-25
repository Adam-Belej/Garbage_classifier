import tensorflow as tf
import utils.network_utils as nu
from models.convolutional_model import Convolutional
import os


def main(input_dir: str,
         test_dir: str,
         num_of_classes: int = 3,
         img_width: int = 512,
         img_height: int = 512):

    classifier = Convolutional(num_of_classes)

    training_set = nu.load_dataset(data_dir=input_dir,
                                   img_width=img_width,
                                   img_height=img_height,
                                   subset="training",
                                   validation_split=classifier.valsplit,
                                   seed=123,
                                   batch_size=classifier.batchsize)
    validation_set = nu.load_dataset(data_dir=input_dir,
                                     img_width=img_width,
                                     img_height=img_height,
                                     subset="validation",
                                     validation_split=classifier.valsplit,
                                     seed=123,
                                     batch_size=classifier.batchsize)

    classifier.load_pretrained_model("/home/adam/Projects/Garbage_Classifier_dataset/test/model.keras")
    classifier.model.summary()
    classifier.train(training_data=training_set,
                     validation_data=validation_set)

    print("History is ", classifier.history)

    classifier.make_graph_from_history()

    classifier.test_accuracy(test_data_dir=test_dir,
                             img_width=img_width,
                             img_height=img_height)
    print(classifier.evaluation)


if __name__ == "__main__":
    input_folder = input("Input directory: ")
    num_classes = int(input("Number of classes: "))
    test_data = str(input("Folder containing test data: "))
    main(input_dir=input_folder,
         num_of_classes=num_classes,
         test_dir=test_data)
