from typing import List, Tuple, Any

from PIL import Image
import os
import typing


def get_all_images_from_dir(folder: str,
                            image_extensions=None) -> [list]:
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    images = []

    for filename in os.listdir(folder):
        if any(filename.endswith(ext) for ext in image_extensions):
            images.append(filename)
    return images


def get_pixel_values(input_image: str) -> [tuple]:
    im = Image.open(input_image)
    pixel_values = tuple(im.getdata())
    return pixel_values


def get_all_categories_from_dir(input_folder) -> [tuple]:
    images = []
    for path in os.listdir(input_folder):
        images.append((os.path.join(input_folder, path), get_all_images_from_dir(os.path.join(input_folder, path))))
    return images

