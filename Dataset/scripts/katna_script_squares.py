from Katna.image import Image
from Katna.writer import ImageCropDiskWriter
import os
import sys.argv

DIRECTORY = '/home/adam/Projects/Garbage_Classifier_dataset/Plast'
DIRECTORY_OUT = '/home/adam/Projects/Garbage_Classifier_dataset_squares'
CROP_ASPECT_RATIO = '1:1'
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
NUM_OF_CROPS = 4
DOWN_SAMPLE_FACTOR = 32


def get_all_images_from_dir(directory):
    images = []
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS):
            images.append(os.path.join(directory, filename))
    return images


def crop_image(image_path):
    image_file_path = os.path.join(directory, filename)
    img_module = Image()
    diskwriter = ImageCropDiskWriter(location=DIRECTORY_OUT)

    crop_list = img_module.crop_image_with_aspect(
        file_path=image_file_path,
        crop_aspect_ratio=CROP_ASPECT_RATIO,
        num_of_crops=NUM_OF_CROPS,
        writer=diskwriter,
        down_sample_factor=DOWN_SAMPLE_FACTOR
    )
    return crop_list


if __name__ == '__main__':
    print("Directory: ", DIRECTORY)

    for filename in get_all_images_from_dir(directory):
        print("Cropping: ", filename)
        crop_image(filename)

