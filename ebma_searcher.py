"""
Estimates the motion between to frame images
 by running an Exhaustive search Block Matching Algorithm (EBMA).

Tried in Python 2.7.5

!! DEPENDENCIES !!
This script depends on the following Python Packages:
- argparse
- scikit-image
- numpy
- matplotlib
"""

import argparse
import os

from skimage.io import imsave

from utils import positive_integer, read_binary_image


if __name__ == "__main__":

    # parse the command line arguments to attributes of 'args'
    parser = argparse.ArgumentParser(description='EBMA searcher. '
                                                 'Estimates the motion between to frame images '
                                                 ' by running an Exhaustive search Block Matching Algorithm (EBMA).')
    parser.add_argument('--target-frame', dest='target_frame_path', required=True, type=str,
                        help='Path to the frame that will be predicted.')
    parser.add_argument('--anchor-frame', dest='anchor_frame_path', required=True, type=str,
                        help='Path to the frame that will be used to predict the target frame.')
    parser.add_argument('--frame-width', dest='frame_width', required=True, type=positive_integer,
                        help='Frame width.')
    parser.add_argument('--frame-height', dest='frame_height', required=True, type=positive_integer,
                        help='Frame height.')
    parser.add_argument('--exponent', dest='p', required=False, type=float, default=1,
                        help='Exponent used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.')
    parser.add_argument('--block-size', dest='block_size', required=True, type=positive_integer,
                        help='Size of the blocks the image will be cut into, in pixels.')
    parser.add_argument('--search-range', dest='search_range', required=True, type=positive_integer,
                        help="Range around the pixel where to search, in pixels.")
    args = parser.parse_args()

    width = args.frame_width
    height = args.frame_height

    # Pixel map of the frames in the [0,255] interval
    target_frame = read_binary_image(args.target_frame_path, (width, height))
    anchor_frame = read_binary_image(args.anchor_frame_path, (width, height))

    # store frames in PNG
    os.system('mkdir -p frames_being_processed')
    imsave('frames_being_processed/target.png', target_frame)
    imsave('frames_being_processed/anchor.png', anchor_frame)