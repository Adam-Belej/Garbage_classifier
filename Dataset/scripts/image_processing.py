from katna_script_squares import get_all_images_from_dir, crop_image_with_aspect_ratio
from ffmpeg_dataset_tools import downscale_image, rotate_image, horizontal_flip_image


path_to_images = str(input("Folder with images: "))
original_images = get_all_images_from_dir(path_to_images)


cropped_images = []
for image in original_images:
    cropped_images.append(crop_image_with_aspect_ratio(image, f"{path_to_images}_cropped"))


downscaled_images = []
for cropped_image in cropped_images:
    downscale_image(cropped_image, f"{cropped_image}_downscaled")
    downscaled_images.append(f"{cropped_image}_downscaled")


if str(input("Augment images? Y/N: ")).lower() == "y":
    for downscaled_image in downscaled_images:
        horizontal_flip_image(downscaled_image, f"{downscaled_image}_flipped")
        for i in range(1, 4):
            rotate_image(downscaled_image, f"{downscaled_image}_rotated_{90*i}")
            rotate_image(f"{downscaled_image}_flipped", f"{downscaled_image}_flipped_rotated_{90*i}")
