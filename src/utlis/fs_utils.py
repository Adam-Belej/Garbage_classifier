import os
import typing


def get_all_images_from_dir(folder: str,
                            image_extensions=None,
                            verbose=False) -> [str]:
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    images = []
    for filename in os.listdir(folder):
        if any(filename.endswith(ext) for ext in image_extensions):
            images.append(filename)
    if verbose:
        print("Found {} images in {}".format(len(images), folder))
    return images
