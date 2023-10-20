from PIL import Image
import os
import typing


def get_all_images_from_dir(folder: str,
                            image_extensions=None) -> [str]:
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    images = []
    for filename in os.listdir(folder):
        if any(filename.endswith(ext) for ext in image_extensions):
            images.append(os.path.join(folder, filename))
    return images


def get_pixel_values(input_image: str):
    im = Image.open(input_image)
    pixel_values = tuple(im.getdata())
    return pixel_values
