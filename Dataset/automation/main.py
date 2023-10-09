#!/bin/python
from ffmpeg_utils import *
from katna_utils import *
import os


def main():

    path_to_images = str(input("Folder with images: "))
    original_images = get_all_images_from_dir(path_to_images)

    cropped_path = f"{path_to_images}_cropped"

    for image in original_images:
        image = os.path.join(path_to_images, image)
        crop_image_with_aspect_ratio(image, f"{path_to_images}_cropped")

    downscaled_path = f"{cropped_path}_downscaled"
    cropped_images = get_all_images_from_dir(cropped_path)

    for cropped_image in cropped_images:
        cropped_image = os.path.join(cropped_path, cropped_image)
        downscaled_image = os.path.join(downscaled_path, cropped_image)
        downscale_image(cropped_image, downscaled_image)

    pass


if __name__ == '__main__':
    main()
