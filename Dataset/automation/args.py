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
                            help="Input directory") # TODO

    return arg_parser.parse_args()

