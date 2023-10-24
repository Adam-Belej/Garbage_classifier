from typing import List, Tuple, Any
import os
import typing
import tensorflow as tf


def load_dataset(data_dir: str,
                 subset: str,
                 validation_split: int = 0.2,
                 img_width: int = 512,
                 img_height: int = 512,
                 batch_size: int = 32,
                 seed: int = 123):
    image_size = (img_height, img_width)
    ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        subset=subset,
        validation_split=validation_split,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed
    )

    autotune = tf.data.AUTOTUNE
    ds = ds.cache().prefetch(buffer_size=autotune)

    return ds
