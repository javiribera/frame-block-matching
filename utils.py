__author__ = 'javiribera'

import argparse

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