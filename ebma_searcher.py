__author__ = 'javiribera'
"""
Estimates the motion between to frame images
 by running an Exhaustive search Block Matching Algorithm (EBMA).

!! DEPENDENCIES !!
This script depends on the following Python Packages:
- argparse
- numpy
- matplotlib
"""

import argparse
from utils import positive_integer
import numpy as np
import matplotlib
from ipdb import set_trace as debug

if __name__ == "__main__":

    # parse the command line arguments to attributes of 'args'
    parser = argparse.ArgumentParser(description='EBMA searcher. '
                                                 'Estimates the motion between to frame images '
                                                 ' by running an Exhaustive search Block Matching Algorithm (EBMA).')
    parser.add_argument('--video', dest='video_path', required=True, type=str,
                        help='Path of the video to analyze.')
    parser.add_argument('--block-size', dest='block_size', required=True, type=positive_integer,
                        help='Size of the blocks the image will be cut into, in pixels.')
    parser.add_argument('--search-range', dest='search_range', required=True, type=positive_integer,
                        help="Range around the pixel where to search, in pixels.")
    args = parser.parse_args()

    debug()