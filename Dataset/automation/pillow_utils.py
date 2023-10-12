import PIL.ImageFile
from PIL import Image
import typing


def downscale_image(input_file: str,
                    output_file: str,
                    width: int = 512,
                    height: int = 512):
    try:
        image = Image.open(input_file)
        image = image.resize((width, height))
        image.save(output_file)

    except PIL.ImageFile.ERRORS as e:
        print(f"Error: {e}")


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


def rotate_image(input_file: str,
                 angle: int = 90):

    for i in range(1, 4):


        new_file = input_file.split(".")
        new_file[0] = f"{new_file[0]}_rotated_{i}"
        output_file = ".".join(new_file)

        new_angle = angle*i

        try:
            image = Image.open(input_file)
            image = image.rotate(new_angle)
            image.save(output_file)

        except PIL.ImageFile.ERRORS as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("reformatting with PIL")
