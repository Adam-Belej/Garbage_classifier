from PIL import Image
import os
import typing


def get_pixel_values(input_image: str):
    im = Image.open(input_image)
    pixel_values = tuple(im.getdata())
    return pixel_values
