from Katna.image import Image
from Katna.writer import ImageCropDiskWriter
import os
import sys.argv
import typing


def get_all_images_from_dir(directory: str,
                            image_extensions=None) -> [str]:
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    images = []
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, filename))
    return images


def crop_image(directory: str,
               directory_out: str,
               crop_aspect_ratio: str = "1:1",
               num_of_crops: int = 1,
               down_sample_factor: int = 32):
    img_module = Image()
    diskwriter = ImageCropDiskWriter(location=directory_out)

    crop_list = img_module.crop_image_with_aspect(
        file_path=directory,
        crop_aspect_ratio=crop_aspect_ratio,
        num_of_crops=num_of_crops,
        writer=diskwriter,
        down_sample_factor=down_sample_factor
    )
    return crop_list


def main():
    directory = input("Directory in: ")
    directory_out = input("Directory out: ")
    print("Directory: ", directory)

    for filename in get_all_images_from_dir(directory):
        print("Cropping: ", filename)
        crop_image(filename, directory_out)


if __name__ == "__main__":
    main()
