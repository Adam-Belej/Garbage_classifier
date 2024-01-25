#!/bin/python
import utils.pillow_utils as pu
import utils.katna_utils as ku
import utils.fs_utils as fs
from utils.console_utils import print_info, print_error, print_warning
import os
import argparse


def get_args():
    arg_parser = argparse.ArgumentParser("jmeno programu", # TODO
                                         description="Popis programu", # TODO
                                         epilog="This is where you might put example usage") # TODO
    arg_parser.add_argument("-v",
                            "--verbose",
                            action="store_true",
                            help="Verbose output") # TODO
    arg_parser.add_argument("-W", "--width",
                            type=int,
                            default=512,
                            help="Width of the image") # TODO
    arg_parser.add_argument("-H", "--height",
                            type=int,
                            default=512,
                            help="Height of the image") # TODO
    arg_parser.add_argument("-o", "--output-dir",
                            type=str,
                            default="./out/",
                            help="Output directory") # TODO
    arg_parser.add_argument("-i", "--input-dir",
                            type=str,
                            default=".",
                            help="Input directory",
                            required=True) # TODO
    arg_parser.add_argument("-e", "--extension",
                            type=str,
                            default="png",
                            help="Extension of the output file") # TODO
    arg_parser.add_argument("-a", "--augmentation",
                            type=bool,
                            default=True,
                            help="Augmentation of images True/False")  # TODO
    return arg_parser.parse_args()


def main(input_dir: str,
         output_dir: str,
         width: int = 512,
         height: int = 512,
         extension: str = "png",
         augmentation: bool = True,
         verbose: bool = False):

    print_info("Starting dataset creation", verbose)
    print_info(f"Input directory: {format(input_dir)}", verbose)

    path_to_images = input_dir
    original_images = fs.get_all_images_from_dir(path_to_images)

    print_info(f"Found {len(original_images)} images", verbose)

    print_info("Starting cropping images")
    for image in original_images:
        if image.endswith('gif'):
            continue
        image = os.path.join(path_to_images, image)
        print_info(f"Cropping {image}")
        ku.crop_image_with_aspect_ratio(image, output_dir)

    edited_images = fs.get_all_images_from_dir(output_dir)

    print_info(f"Downscaling images to {width}x{height}", verbose)
    for cropped_image in edited_images:
        downscaled_image = os.path.join(output_dir, cropped_image)
        downscaled_image = pu.downscale_image(downscaled_image, extension=extension, width=width, height=height)

        if augmentation:
            pu.horizontal_flip_image(downscaled_image)

    if not augmentation:
        print_info("Skipping augmentation", verbose)
        return
    edited_images = fs.get_all_images_from_dir(output_dir)

    print_info("Augmenting images", verbose)
    for edited_image in edited_images:
        edited_image = os.path.join(output_dir, edited_image)
        pu.rotate_image(edited_image)


if __name__ == '__main__':
    args = get_args()
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         width=args.width,
         height=args.height,
         extension=args.extension,
         augmentation=args.augmentation,
         verbose=args.verbose)
    print_info("Done", args.verbose)

