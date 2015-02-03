"""
Estimates the motion between to frame images
 by running an Exhaustive search Block Matching Algorithm (EBMA).
Minimizes the norm of the Displaced Frame Difference (DFD).

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
import itertools

import numpy as np


# import timeit

from skimage.io import imsave

from utils import positive_integer, read_binary_image, subarray, show_quiver


def main():
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
    parser.add_argument('--norm', dest='norm', required=False, type=float, default=1,
                        help='Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.')
    parser.add_argument('--block-size', dest='block_size', required=True, type=positive_integer,
                        help='Size of the blocks the image will be cut into, in pixels.')
    parser.add_argument('--search-range', dest='search_range', required=True, type=positive_integer,
                        help="Range around the pixel where to search, in pixels.")
    args = parser.parse_args()

    # Pixel map of the frames in the [0,255] interval
    target_frm = read_binary_image(args.target_frame_path, (args.frame_height, args.frame_width))
    anchor_frm = read_binary_image(args.anchor_frame_path, (args.frame_height, args.frame_width))

    # store frames in PNG for our records
    os.system('mkdir -p frames_being_processed')
    imsave('frames_being_processed/target.png', target_frm)
    imsave('frames_being_processed/anchor.png', anchor_frm)

    ebma = EBMA_searcher(shape=(args.frame_height, args.frame_width),
                         N=args.block_size,
                         R=args.search_range,
                         p=args.norm)

    predicted_frame, motion_field = \
        ebma.run(anchor_frame=anchor_frm,
                 target_frame=target_frm)

    # store predicted frame
    imsave('frames_being_processed/predicted_target.png', predicted_frame)

    motion_field_x = motion_field[:, :, 0]
    motion_field_y = motion_field[:, :, 1]

    # show motion field
    show_quiver(motion_field_x, motion_field_y[::-1])


class EBMA_searcher():
    """
    Estimates the motion between to frame images
     by running an Exhaustive search Block Matching Algorithm (EBMA).
    Minimizes the norm of the Displaced Frame Difference (DFD).
    """

    def __init__(self, shape, N, R, p=1):
        """
        :param shape: Shape of all the frames in pixels, as (height, width).
        :param N: Size of the blocks the image will be cut into, in pixels.
        :param R: Range around the pixel where to search, in pixels.
        :param p: Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.
        """

        self.height = shape[0]
        self.width = shape[1]
        self.N = N
        self.R = R
        self.p = p

    def run(self, anchor_frame, target_frame):
        """
        Run!
        :param anchor_frame: Frame that will be used to predict the target frame.
        :param target_frame: Frame that will be predicted.
        :return: A tuple consisting of the predicted image and the motion field.
        """

        height = self.height
        width = self.width
        N = self.N
        R = self.R
        p = self.p

        # predicted frame. target_frame is predicted from anchor_frame
        predicted_frame = np.empty(((height, width)), dtype=np.uint8)

        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((int(height / N), int(width / N), 2))

        # loop through every NxN block in the image
        for (blk_row, blk_col) in itertools.product(xrange(0, height - (N - 1), N),
                                                    xrange(0, width - (N - 1), N)):

            # block whose match will be searched in the anchor frame
            blk = target_frame[blk_row:blk_row + N, blk_col:blk_col + N]

            # minimum norm of the DFD norm found so far
            dfd_n_min = np.infty

            # search which block in a surrounding RxR region minimizes the norm of the DFD. Blocks overlap.
            for (r_col, r_row) in itertools.product(range(-R, R),
                                                    range(-R, R)):
                # candidate block upper left vertex and lower right vertex position as (row, col)
                up_l_candidate_blk = (blk_row + r_row, blk_col + r_col)
                low_r_candidate_blk = (blk_row + r_row + N - 1, blk_col + r_col + N - 1)

                # don't search outside the anchor frame
                if up_l_candidate_blk[0] < 0 or up_l_candidate_blk[1] < 0 or \
                                low_r_candidate_blk[0] > height - 1 or low_r_candidate_blk[1] > width - 1:
                    continue

                # the candidate block may fall outside the anchor frame
                candidate_blk = subarray(anchor_frame, up_l_candidate_blk, low_r_candidate_blk)
                assert candidate_blk.shape == (N, N)

                dfd = np.array(candidate_blk, dtype=np.float64) - np.array(blk, dtype=np.float64)

                candidate_dfd_norm = np.linalg.norm(dfd, ord=p)

                # a better matching block has been found. Save it and its displacement
                if candidate_dfd_norm < dfd_n_min:
                    dfd_n_min = candidate_dfd_norm
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            # construct the predicted image with the block that matches this block
            predicted_frame[blk_row:blk_row + N, blk_col:blk_col + N] = matching_blk

            # displacement of this block in each direction
            print str((blk_row / N, blk_col / N)) + '---' + str((dx, dy))

            motion_field[blk_row / N, blk_col / N, 1] = dx
            motion_field[blk_row / N, blk_col / N, 0] = dy

        return predicted_frame, motion_field


if __name__ == "__main__":
    main()

    # tictoc = timeit.timeit(main, number=1)
    # print 'Time to run the whole main(): %s' %tictoc