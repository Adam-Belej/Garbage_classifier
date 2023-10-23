import PIL.ImageFile
from PIL import Image
import os
import typing


def downscale_image(input_file: str,
                    extension: str = "png",
                    width: int = 512,
                    height: int = 512):
    new_file = input_file.split(".")
    new_file[1] = extension
    output_file = ".".join(new_file)
    try:
        image = Image.open(input_file)
        image = image.convert("RGB")
        image = image.resize((width, height))
        image.save(output_file)
        os.remove(input_file)

    except PIL.ImageFile.ERRORS as e:
        print(f"Error: {e}")
        print(f"Error while downscaling {input_file}")
        exit()

    return output_file


def horizontal_flip_image(input_file: str):
    new_file = input_file.split(".")
    new_file[0] = f"{new_file[0]}_flipped"
    output_file = ".".join(new_file)

    try:
        image = Image.open(input_file)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image.save(output_file)

    except PIL.ImageFile.ERRORS as e:
        print(f"Error: {e}")
        print(f"Error while flipping {input_file}")
        exit()


def rotate_image(input_file: str,
                 angle: int = 90):
    for i in range(1, 4):

        new_file = input_file.split(".")
        new_file[0] = f"{new_file[0]}_rotated_{i}"
        output_file = ".".join(new_file)

        new_angle = angle * i

        try:
            image = Image.open(input_file)
            image = image.rotate(new_angle)
            image.save(output_file)

        except PIL.ImageFile.ERRORS as e:
            print(f"Error: {e}")
            print(f"Error while rotating {input_file}")
            exit()


if __name__ == "__main__":
    print("reformatting with PIL")
