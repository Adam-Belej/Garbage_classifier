import argparse


def get_args():
    arg_parser = argparse.ArgumentParser("jmeno programu", # TODO
                                         description="Popis programu", # TODO
                                         epilog="This is where you might put example usage") # TODO

    arg_parser.add_argument("-v",
                            "--verbose",
                            action="store_true",
                            help="Verbose output") # TODO

    arg_parser.add_argument("-w", "--width",
                            type=int,
                            default=512,
                            help="Width of the image") # TODO

    arg_parser.add_argument("-H", "--height",
                            type=int,
                            default=512,
                            help="Height of the image") # TODO

    arg_parser.add_argument("-o", "--output-dir",
                            type=str,
                            default="./out/",
                            help="Output directory") # TODO

    arg_parser.add_argument("-i", "--input-dir",
                            type=str,
                            default=".",
                            help="Input directory",
                            required=True) # TODO

    arg_parser.add_argument("-e", "--extension",
                            type=str,
                            default="png",
                            help="Extension of the output file") # TODO

    arg_parser.add_argument("-a", "--augmentation",
                            type=bool,
                            default=True,
                            help="Augmentation of images True/False")  # TODO

    return arg_parser.parse_args()


def print_args(args):
    print("verbose:", args.verbose)
    print("width:", args.width)
    print("height:", args.height)
    print("output_dir:", args.output_dir)
    print("input_dir:", args.input_dir)
    print("extension:", args.extension)
    print("augmentation:", args.augmentation)
