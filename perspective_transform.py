"""
Author: YWiyogo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

DEBUG = 0


class PerspectiveTransform:
    """Class of perspective transform"""

    def __init__(self):
        """Constructor"""
        # Calibrate the trapezoid first with test_images/straight_lines1.jpg
        # The trapezoid shall not to long, since it will affects the detection
        # for the curve
        self.src = np.float32([[278, 670],
                               [588, 455],  # [567, 470],
                               [693, 455],  # [715, 470],
                               [1030, 670]])

        self.dst = np.float32([[280, 710],
                               [280, 20],
                               [1030, 20],
                               [1030, 710]])
        self.M = 0
        self.Minv = 0

    def search_start_point(self, thres_img):
        """Finding the lower point of left line. Currently not used"""
        # visualize the histogram of the lowest image area of thres S channel
        histogram = np.sum(thres_img[650:700, :, 1], axis=0)
        if DEBUG:
            plt.plot(histogram)
            # add the distribution of both channels
            plt.show()

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_start = np.argmax(histogram[:midpoint], axis=0)
        # if there is an activation of the line, replace the x start point
        if histogram[leftx_start] > 10:
            self.src[0, 0] = leftx_start
            if DEBUG:
                print("count: {}".format(histogram[leftx_start]))
                print("start left: {}".format(self.src))

    def do_perpectivetransform(self):
        """Run perspective transform"""
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = np.linalg.inv(self.M)

    def warp(self, img):
        """Warping image"""
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        """Warping image"""
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def visualize_transform(self, src_img):
        """Visualize the transformation line"""
        if DEBUG:
            dst_img = cv2.warpPerspective(src_img, self.M, (src_img.shape[1],
                                                            src_img.shape[0]),
                                          flags=cv2.INTER_LINEAR)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
            f.tight_layout()
            ax1.set_title("source")
            ax2.set_title("destination")
            ax1.plot(self.src[:, 0], self.src[:, 1], "r")
            ax2.plot(self.dst[:, 0], self.dst[:, 1], "r")
            ax1.imshow(src_img)
            ax2.imshow(dst_img)
            plt.show()
