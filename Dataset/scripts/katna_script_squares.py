from Katna.image import Image
from Katna.writer import ImageCropDiskWriter
import os

directory = '/home/adam/Projects/Garbage_Classifier_dataset/Plast'
img_module = Image()
diskwriter = ImageCropDiskWriter(location="/home/adam/Projects/Garbage_Classifier_dataset_squares")

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    image_file_path = f
    crop_aspect_ratio = '1:1'

    crop_list = img_module.crop_image_with_aspect(
        file_path=image_file_path,
        crop_aspect_ratio=crop_aspect_ratio,
        num_of_crops=1,
        writer=diskwriter,
        down_sample_factor=32)
