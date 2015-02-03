import argparse

import matplotlib.pyplot as plt
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
    pixels = pixels.reshape((height, width))

    return pixels


def show_gray_img(asd):
    plt.imshow(asd, cmap='gray')
    plt.show()


def show_quiver(x_component_arrows, y_components_arrows):
    plt.quiver(x_component_arrows, y_components_arrows)
    plt.show()


def subarray(array, (upper_left_pix_row, upper_left_pix_col), (lower_right_pix_row, lower_right_pix_col)):
    """
    Return a subarray containing the pixels delimited by the pixels between upper_left_pix and lower_right.
    If asked for pixels outside the image boundary, such pixels have value 0.
    """

    if upper_left_pix_row > lower_right_pix_row or upper_left_pix_col > lower_right_pix_col:
        raise ValueError('coordinates of the subarray should correspond to a meaningful rectangle')

    orig_array = np.array(array)

    num_rows = lower_right_pix_row - upper_left_pix_row + 1
    num_cols = lower_right_pix_col - upper_left_pix_col + 1

    subarr = np.zeros((num_rows, num_cols), dtype=orig_array.dtype)

    # zoomed outside the original image
    if lower_right_pix_col < 0 or lower_right_pix_row < 0 or \
                    upper_left_pix_col > orig_array.shape[1] - 1 or upper_left_pix_row > orig_array.shape[0] - 1:
        return subarr

    # region of the original image that is inside the desired region
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_o_1, i_o_1)              |    |
    # |   |             (j_o_2, i_o_2) |    |
    # |___|____________________________|    |
    # |                                 |
    # |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_o_1 = 0
    else:
        i_o_1 = upper_left_pix_col
    if upper_left_pix_row < 0:
        j_o_1 = 0
    else:
        j_o_1 = upper_left_pix_row

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_o_2 = orig_array.shape[1] - 1
    else:
        i_o_2 = lower_right_pix_col
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_o_2 = orig_array.shape[0] - 1
    else:
        j_o_2 = lower_right_pix_row


    # region of the final image that is inside the original image, and whose content will be taken from the orig im
    # (i = col, j=row)
    #  _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_f_1, i_f_1)              |    |
    # |   |             (j_f_2, i_f_2) |    |
    # |___|____________________________|    |
    #     |                                 |
    #     |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_f_1 = -upper_left_pix_col
    else:
        i_f_1 = 0
    if upper_left_pix_row < 0:
        j_f_1 = -upper_left_pix_row
    else:
        j_f_1 = 0

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_f_2 = (orig_array.shape[1] - 1) - upper_left_pix_col
    else:
        i_f_2 = num_cols - 1
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_f_2 = (orig_array.shape[0] - 1) - upper_left_pix_row
    else:
        j_f_2 = num_rows - 1

    subarr[j_f_1:j_f_2 + 1, i_f_1:i_f_2 + 1] = orig_array[j_o_1:j_o_2 + 1, i_o_1:i_o_2 + 1]

    return subarr
