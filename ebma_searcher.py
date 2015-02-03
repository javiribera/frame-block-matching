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

    # size of all the frames
    width = args.frame_width
    height = args.frame_height

    # Pixel map of the frames in the [0,255] interval
    target_frame = read_binary_image(args.target_frame_path, (height, width))
    anchor_frame = read_binary_image(args.anchor_frame_path, (height, width))

    # store frames in PNG for our records
    os.system('mkdir -p frames_being_processed')
    imsave('frames_being_processed/target.png', target_frame)
    imsave('frames_being_processed/anchor.png', anchor_frame)

    # block size
    N = args.block_size

    # search range
    R = args.search_range

    # the p-norm of the DFD will be minimized
    p = args.norm

    # predicted frame. target_frame is predicted from anchor_frame
    predicted_frame = np.empty(((height, width)), dtype=np.uint8)

    # optical flow consisting in the displacement of each block in vertical and horizontal
    optical_flow_x = np.empty((int(height / N), int(width / N)))
    optical_flow_y = np.empty((int(height / N), int(width / N)))

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

            # # don't search outside the anchor frame
            # if up_l_candidate_blk[0] < 0 or up_l_candidate_blk[1] < 0 \
            # or low_r_candidate_blk[0] > height - 1 or low_r_candidate_blk[1] > width -1:
            # continue

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

        optical_flow_x[blk_row / N, blk_col / N] = dx
        optical_flow_y[blk_row / N, blk_col / N] = dy

    # store predicted frame
    imsave('frames_being_processed/predicted_target.png', predicted_frame)

    # show optical flow
    show_quiver(optical_flow_x[::-1], optical_flow_y[::-1])
    pass


if __name__ == "__main__":
    main()

    # tictoc = timeit.timeit(main, number=1)
    # print 'Time to run the whole main(): %s' %tictoc