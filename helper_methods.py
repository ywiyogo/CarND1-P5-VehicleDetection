"""
Author: YWiyogo
Description: Helper function for P3
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog

# Analyzing the training datasets

DEBUG = 1

def visualize_img(img, title, cmap):
    fig = plt.figure()
    plt.imshow(img)
    plt.show()

def visualize_imgs(imglist, titlelist, cmaplist):
    """Visualize list of image"""
    if DEBUG:
        rows = int(len(imglist) / 2) + (len(imglist) % 2 > 0)
        f, axarr = plt.subplots(rows, 2, figsize=(10, 8))
        f.tight_layout()
        i = 0
        j = 0
        for idx, img in enumerate(imglist):
            if rows < 2:
                axis = axarr[i]
                i = i + 1
            else:
                axis = axarr[i, j]
                if j < axarr.shape[1] - 1:
                    j = j + 1
                else:
                    i = i + 1
                    j = 0
            axis.set_title(titlelist[idx])
            if cmaplist[idx] == 1:
                axis.imshow(img, cmap="gray")
            else:
                axis.imshow(img)
        plt.show()


def all_color_thresholding(img):
    """All color"""
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    white_3 = cv2.inRange(img, (200,200,200), (255,255,255))

    bit_layer = yellow | white | white_2 | white_3

    return bit_layer


def s_channel_thresholding(img, thresh_min=170, thresh_max=255):
    """Convert image to HLS and return the S channel and its thresholding"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    return s_channel, binary_s


def sobel_color_threshold(img, orient='x', thresh_min=25, thresh_max=100, debug=False):
    """Perform binary thresholding based on Sobel operator and S channel"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value

    kernelsize = 3
    if orient == 'x':
        sobel_gray = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernelsize))
    if orient == 'y':
        sobel_gray = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernelsize))
    abs_sobel_gray = np.absolute(sobel_gray)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel_gray / np.max(abs_sobel_gray))
    # Note: It's not entirely necessary to convert to 8-bit (range from 0 to
    # 255) but in practice, it can be useful in the event that you've written
    # a function to apply a particular threshold, and you want it to work the
    # same on input images of different scales, like jpg vs. png.

    # Create a copy and apply the binary threshold
    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel >= thresh_min) &
                 (scaled_sobel <= thresh_max)] = 1
    # call S channel
    s_channel, binary_s = s_channel_thresholding(img, 190, 255)

    color_thres = all_color_thresholding(img)
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
    # It might be beneficial to replace this channel with something else.
    combine_binary = np.dstack(
        (np.zeros_like(color_thres), color_thres, color_thres))
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    if debug: visualize_imgs((img, hls, s_channel, color_thres, binary_s, combine_binary),
                  ("RGB", "hls", "s_channel", "color_thres",
                   "binary_s", "combine_binary"),
                  (0, 0, 0, 1, 1, 1))
    return combine_binary

# ----------------------------------
# New Helper method for P5
# ----------------------------------


def convert_rgb_to(rgbimg, cspace):
    """Convert RGB img to defined color space"""
    if cspace == 'HLS':
        img = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV)
    elif cspace == 'YCrCb':
        img = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2LUV)
    elif cspace == 'HSV':
        img = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        img = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2YUV)
    elif cspace == 'LUV':
        img = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2YCrCb)
    return img


def show_histogram(data, title="Histogram of the datasets"):
    """
    Plotting histogram
    """
    fig_hist = plt.figure(figsize=(15, 8))
    ax = fig_hist.add_subplot(111)
    ax.hist(data, rwidth=0.1, align="mid", zorder=3)
    ax.yaxis.grid(True, linestyle='--', zorder=0)
    ax.set_ylabel('Occurrences')
    ax.set_xlabel('Value')
    ax.set_title(title)
    plt.show()

def show_barplot(data, title="Features"):
    fig = plt.figure()
    if (isinstance(data, np.ndarray)):
        ind= np.arange(data.shape[0])
        print(ind)
        print(data)
        plt.title(title)
        # plt.ylim(min(data),max(data))
        plt.bar(ind, data)

    else:
        ind= np.arange(data[0].shape[0])
        plt.title("Features")
        axis=[]
        legends=title
        for i, arr in enumerate(data):
            ax = plt.bar(ind, arr)
            axis.append(ax)
        plt.legend(axis,legends)

    plt.show()

