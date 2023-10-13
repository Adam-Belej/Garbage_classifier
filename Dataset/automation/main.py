#!/bin/python
import pillow_utils as pu
import katna_utils as ku
import fs_utils as fs
from args import get_args, print_args
import os


def main(input_dir: str,
         output_dir: str,
         width: int = 512,
         height: int = 512,
         extension: str = "png",
         augmentation: bool = True):

    path_to_images = input_dir
    original_images = fs.get_all_images_from_dir(path_to_images)

    for image in original_images:
        image = os.path.join(path_to_images, image)
        ku.crop_image_with_aspect_ratio(image, output_dir)

    edited_images = fs.get_all_images_from_dir(output_dir)

    for cropped_image in edited_images:
        downscaled_image = os.path.join(output_dir, cropped_image)
        downscaled_image = pu.downscale_image(downscaled_image, extension=extension, width=width, height=height)

        if augmentation:
            pu.horizontal_flip_image(downscaled_image)

    if not augmentation:
        return
    edited_images = fs.get_all_images_from_dir(output_dir)

    for edited_image in edited_images:
        edited_image = os.path.join(output_dir, edited_image)
        pu.rotate_image(edited_image)


if __name__ == '__main__':
    args = get_args()
    print_args(args)
    main(args)
