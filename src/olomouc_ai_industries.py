import tensorflow as tf
import utils.network_utils as nu
from models.Alfonzo import Alfonzo
import os


def main(input_dir: str,
         num_of_classes: int = 3,
         img_width: int = 512,
         img_height: int = 512):
    training_set = nu.load_dataset(data_dir=input_dir,
                                   img_width=img_width,
                                   img_height=img_height)

    alfonz = Alfonzo(num_of_classes)

    alfonz.train(training_set)


if __name__ == "__main__":
    input_folder = input("Input directory: ")
    num_classes = int(input("Number of classes: "))
    main(input_dir=input_folder,
         num_of_classes=num_classes)
