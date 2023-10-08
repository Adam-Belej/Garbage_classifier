import os
from katna_utils import get_all_images_from_dir, crop_image_with_aspect_ratio
from ffmpeg_utils import downscale_image, rotate_image, horizontal_flip_image


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


# f str(input("Augment images? Y/N: ")).lower() == "y":
#     for downscaled_image in downscaled_images:
#         horizontal_flip_image(downscaled_image, f"{downscaled_image}_flipped")
#         for i in range(1, 4):
#             rotate_image(downscaled_image, f"{downscaled_image}_rotated_{90*i}")
#             rotate_image(f"{downscaled_image}_flipped", f"{downscaled_image}_flipped_rotated_{90*i}")
