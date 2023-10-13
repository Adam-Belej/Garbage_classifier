#!/bin/python
import pillow_utils as pu
import katna_utils as ku
import fs_utils as fs
from args import get_args, print_args
import os


def main(args):
    augmentation = str(input("Augment images? Y/n")).lower()
    path_to_images = str(input("Folder with images: "))
    original_images = fs.get_all_images_from_dir(path_to_images)

    edited_path = f"{path_to_images}_edited"

    for image in original_images:
        image = os.path.join(path_to_images, image)
        ku.crop_image_with_aspect_ratio(image, edited_path)

    edited_images = fs.get_all_images_from_dir(edited_path)

    for cropped_image in edited_images:
        downscaled_image = os.path.join(edited_path, cropped_image)
        downscaled_image = pu.downscale_image(downscaled_image)

        if augmentation == "y":
            pu.horizontal_flip_image(downscaled_image)

    if augmentation == "y":
        edited_images = fs.get_all_images_from_dir(edited_path)

        for edited_image in edited_images:
            edited_image = os.path.join(edited_path, edited_image)
            pu.rotate_image(edited_image)


if __name__ == '__main__':
    args = get_args()
    print_args(args)
    main(args)
