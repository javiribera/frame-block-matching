__author__ = 'javiribera'

import argparse

import numpy as np


def positive_integer(number):
    """
    Convert a number to an positive integer if possible.
    :param number: The number to be converted to a positive integer.
    :return: The positive integer
    :raise: argparse.ArgumentTypeError if not able to do the conversion
    """
    try:
        integ = int(number)
        if integ >= 0:
            return integ
        else:
            raise argparse.ArgumentTypeError('%s is not a positive integer' % number)
    except ValueError:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % number)


def read_binary_image(path, (height, width)):
    """
    Read binary gray-scale image. Each pixel is considered stored in contiguous bytes.
    :param path: Path to the image to read.
    :param (height, width): Height and width of the image.
    :return: The numpy array containing the resulting image.
    """

    # open file in binary format
    with open(path, "rb") as fd:
        whole_file = fd.read(height * width)
    pixels = np.array([ord(byte) for byte in whole_file], dtype=np.uint8)  # get ASCII value of each byte of the file
    pixels = pixels.reshape((width, height))

    return pixels