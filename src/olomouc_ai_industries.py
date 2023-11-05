import tensorflow as tf
import utils.network_utils as nu
from models.Alfonzo import Alfonzo
import os


def main(input_dir: str,
         test_image: str,
         num_of_classes: int = 3,
         img_width: int = 512,
         img_height: int = 512):
    training_set = nu.load_dataset(data_dir=input_dir,
                                   img_width=img_width,
                                   img_height=img_height,
                                   subset="training")
    validation_set = nu.load_dataset(data_dir=input_dir,
                                     img_width=img_width,
                                     img_height=img_height,
                                     subset="validation")

    alfonz = Alfonzo(num_of_classes)

    history = alfonz.train(training_data=training_set,
                           validation_data=validation_set)

    print("History is ", history.history)

    score = alfonz.classify(data=test_image,
                            height=img_height,
                            width=img_width)
    print("Score is ", score)


if __name__ == "__main__":
    input_folder = input("Input directory: ")
    num_classes = int(input("Number of classes: "))
    test_img = str(input("Image to classify: "))
    main(input_dir=input_folder,
         num_of_classes=num_classes,
         test_image=test_img)
