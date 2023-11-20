from typing import List, Tuple, Any
import os
import typing
import tensorflow as tf


def load_dataset(data_dir: str,
                 subset: str,
                 validation_split: int = None,
                 img_width: int = 512,
                 img_height: int = 512,
                 batch_size: int = 32,
                 seed: int = 123):
    image_size = (img_height, img_width)
    if validation_split is not None:
        ds = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir,
            subset=subset,
            validation_split=validation_split,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed
        )
    else:
        ds = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir,
            image_size=image_size,
            batch_size=batch_size,
        )

    return ds


def pretrained_model(directory: str):
    model = tf.saved_model.load(directory)
    return model



def predict_image(img_dir: str, img_width: int, img_height: int, model):
    img = tf.keras.utils.load_img(path=img_dir,
                                  target_size=(img_height, img_width))
    model.predict(img)