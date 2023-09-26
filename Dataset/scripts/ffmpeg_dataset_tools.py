import ffmpeg


def resize_image(input_file, output_file):
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file, vf='scale=512:512')

    try:
        ffmpeg.run(output_stream)
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")


def flip_image(input_file, output_file):
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file, vf='hflip')

    try:
        ffmpeg.run(output_stream)
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")


def rotate_image(input_file, output_file, angle):
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file, vf=f'rotate={angle}')

    try:
        ffmpeg.run(output_stream)
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr}")


if __name__ == "__main__":
    print("reformatting with ffmpeg")
