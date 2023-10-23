from Katna.image import Image
from Katna.writer import ImageCropDiskWriter
import typing


def crop_image_with_aspect_ratio(
        directory: str,
        folder_out: str,
        crop_aspect_ratio: str = "1:1",
        num_of_crops: int = 1,
        down_sample_factor: int = 32):
    img_module = Image()
    diskwriter = ImageCropDiskWriter(location=folder_out)

    crop_list = img_module.crop_image_with_aspect(
        file_path=directory,
        crop_aspect_ratio=crop_aspect_ratio,
        num_of_crops=num_of_crops,
        writer=diskwriter,
        down_sample_factor=down_sample_factor
    )
    return crop_list


