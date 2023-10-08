import ffmpeg
import typing


def downscale_image(input_file: str,
                    output_file: str,
                    width: int = 512,
                    height: int = 512):
    try:
        (
            ffmpeg
            .input(input_file)
            .filter("scale", width, height)
            .output(output_file)
            .run()
        )
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")


def horizontal_flip_image(input_file: str,
                          output_file: str):
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file, vf='hflip')

    try:
        ffmpeg.run(output_stream)
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")


def rotate_image(input_file: str,
                 output_file: str,
                 angle: int = 90):
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file, vf=f'rotate={angle}')

    try:
        ffmpeg.run(output_stream)
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")


if __name__ == "__main__":
    print("reformatting with ffmpeg")
