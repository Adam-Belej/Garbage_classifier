import tensorflow as tf
import utils.network_utils as nu
from models.Alfonzo import Alfonzo
import os


def main(input_dir: str,
         num_of_classes: int = 3,
         img_width: int = 512,
         img_height: int = 512):

    alfonz = Alfonzo(num_of_classes)

    training_set = nu.load_dataset(data_dir=input_dir,
                                   img_width=img_width,
                                   img_height=img_height,
                                   subset="training",
                                   validation_split=alfonz.valsplit,
                                   seed=123,
                                   batch_size=alfonz.batchsize)
    validation_set = nu.load_dataset(data_dir=input_dir,
                                     img_width=img_width,
                                     img_height=img_height,
                                     subset="validation",
                                     validation_split=alfonz.valsplit,
                                     seed=123,
                                     batch_size=alfonz.batchsize)

    alfonz.train(training_data=training_set,
                 validation_data=validation_set)

    print("History is ", alfonz.history)

    alfonz.make_graph_from_history("/home/adam/Projects/Garbage_Classifier_dataset/test/graph1.png")

    alfonz.export_model("/home/adam/Projects/Garbage_Classifier_dataset/test")


if __name__ == "__main__":
    input_folder = input("Input directory: ")
    num_classes = int(input("Number of classes: "))
    main(input_dir=input_folder,
         num_of_classes=num_classes)
