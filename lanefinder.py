"""
Author: Y.Wiyogo
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEBUG = 0


class LaneFinder:
    """Lane Finder Class"""

    def __init__(self):
        """Constructor"""
        self.plot_y = 0
        # The x points of the both lines
        self.llane_x = []
        self.rlane_x = []
        # Array of Polynomial Coefficients
        self.llane_coeffis = []
        self.rlane_coeffis = []
        # initial x-axis base for sliding windows
        self.leftx_base = 280
        self.rightx_base = 1030

        self.lrad_curve = 0
        self.rrad_curve = 0
        # Was
        self.left_detected = False
        self.right_detected = False

        self.y_m_per_pix = 30 / 720  # meters per pixel in y dimension
        self.x_m_per_pix = 3.7 / 700  # meters per pixel in x dimension

        self.veh_pos = 0

    def detect_lanes(self, binwarped_img):
        """Calculate histogram of a binary warped image"""
        half_y = int(binwarped_img.shape[0] / 2)
        # binary warped image has 3 channels
        histogram = np.sum(binwarped_img[half_y:, :, 1:], axis=0)
        plt.plot(histogram)
        # add the distribution of both channels
        histogram = histogram[:, 0] + histogram[:, 1]

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binwarped_img, binwarped_img, binwarped_img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint], axis=0)
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if abs(self.leftx_base - leftx_base) < 310:
            self.leftx_base = leftx_base
        else:
            self.left_detected = False
            print("Warning left x base outlier found, diff:{}".format(
                abs(self.leftx_base - leftx_base)))
        if abs(self.rightx_base - rightx_base) < 310:
            self.rightx_base = rightx_base
        else:
            self.right_detected = False
            print("Warning right x base outlier found, diff:{}".format(
                abs(self.rightx_base - rightx_base)))

        if DEBUG:
            plt.plot(histogram)
            plt.title("Histogram of the binary image")
            plt.show()
            print("Histogram shape: ", histogram.shape)
            print("midpoint: ", midpoint)
            print("Left base: ", self.leftx_base)
            print("Right base: ", self.rightx_base)

        # --------------Sliding windows ------------------------
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 130    # due to test4 image, I enlarged the window width
        minpix = 70  # Set minimum number of pixels found to recenter window
        min_detected_win = int(nwindows * 0.2)
        detected_win_count = [0, 0]  # 8left,right)]
        # Set height of windows
        window_height = np.int(binwarped_img.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binwarped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        lwindows = []
        rwindows = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binwarped_img.shape[0] - (window + 1) * window_height
            win_y_high = binwarped_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # append windows for visualization
            lwindows.append((win_xleft_low, win_y_high))
            rwindows.append((win_xright_low, win_y_high))
            # Draw the windows on the visualization image
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window
            # on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                detected_win_count[0] = detected_win_count[0] + 1
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                detected_win_count[1] = detected_win_count[1] + 1

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(lefty) < 1 or len(righty) < 1:
            print("Error: No index found on lane: {}".format(left_lane_inds))
            return self.llane_x, self.rlane_x
        # ------------ Line Fitting -----------------------
        # Fit a second order polynomial to each,
        # polyfit return ndarray of coefficients
        self.llane_coeffis = np.polyfit(lefty, leftx, 2)
        self.rlane_coeffis = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting, Ax^2 + By + C
        self.plot_y = np.linspace(0, binwarped_img.shape[0] - 1,
                                  binwarped_img.shape[0])

        # Detecting low detection windows
        if detected_win_count[0] >= min_detected_win:
            self.left_detected = True
            self.llane_x = (self.llane_coeffis[0] * self.plot_y**2 +
                            self.llane_coeffis[1] * self.plot_y +
                            self.llane_coeffis[2])
        else:
            self.left_detected = False
            print("Warning left detected windows is too low: {} < {}".format(
                detected_win_count[0], min_detected_win))
        if detected_win_count[1] >= min_detected_win:
            self.right_detected = True
            self.rlane_x = (self.rlane_coeffis[0] * self.plot_y**2 +
                            self.rlane_coeffis[1] * self.plot_y +
                            self.rlane_coeffis[2])
        else:
            self.right_detected = False
            print("Warning right detected windows is too low: {} < {}".format(
                detected_win_count[1], min_detected_win))

        if DEBUG:
            print("Out img: ", out_img.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(binwarped_img)
            if len(self.llane_x) == len(self.plot_y):
                plt.plot(self.llane_x, self.plot_y, color='yellow')
            if len(self.rlane_x) == len(self.plot_y):
                plt.plot(self.rlane_x, self.plot_y, color='yellow')
            for idx in range(len(lwindows)):
                ax.add_patch(Rectangle(lwindows[idx], 2 * margin, window_height,
                             fill=False, alpha=1, color="red"))
                ax.add_patch(Rectangle(rwindows[idx], 2 * margin, window_height,
                             fill=False, alpha=1, color="red"))
            ax.set_title("Sliding windows and The Fitted Lane Line")
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.title("Fitted Lane Line")
            plt.show()

        return self.llane_x, self.rlane_x

    def calc_radcurvature(self):
        """Determine the curvature of the lane."""
        # Define conversions in x and y from pixels space to meters
        if ((len(self.llane_x) != len(self.plot_y)) and
           (len(self.rlane_x) != len(self.plot_y))):
            print("Warning one of the line cannot be detected!")
            return

        if len(self.llane_x) < 1 :
            return
        # Fit new polynomials to x,y in world space
        llane_w_coeffis = np.polyfit(self.plot_y * self.y_m_per_pix,
                                     self.llane_x * self.x_m_per_pix, 2)
        rlane_w_coeffis = np.polyfit(self.plot_y * self.y_m_per_pix,
                                     self.rlane_x * self.y_m_per_pix, 2)
        # Calculate the new radius of curvature
        # R_curve = (1 + (2Ay+B)^2 )^1.5 / |2A|
        y_max = np.max(self.plot_y)
        left_dx_dy = (2 * llane_w_coeffis[0] * y_max * self.y_m_per_pix +
                      llane_w_coeffis[1])
        right_dx_dy = (2 * rlane_w_coeffis[0] * y_max * self.y_m_per_pix +
                       rlane_w_coeffis[1])
        self.lrad_curve = (((1 + left_dx_dy ** 2) ** 1.5) /
                           np.absolute(2 * llane_w_coeffis[0]))
        self.rrad_curve = (((1 + right_dx_dy ** 2) ** 1.5) /
                           np.absolute(2 * rlane_w_coeffis[0]))
        # Now our radius of curvature is in meters
        if DEBUG:
            print("Radius curvature: left: {} m, right: {} m".format(
                self.lrad_curve, self.rrad_curve))

    def calc_vehicle_position(self, undist_img):
        """Determine the vehicle position in respect to center lane"""
        y_pos = undist_img.shape[0] - 1
        # llane_x = (self.llane_coeffis[0] * (y_pos - 1) ** 2 +
        #            self.llane_coeffis[1] * (y_pos - 1) +
        #            self.llane_coeffis[2])
        # rlane_x = (self.rlane_coeffis[0] * (y_pos - 1) ** 2 +
        #            self.rlane_coeffis[1] * (y_pos - 1) +
        #            self.rlane_coeffis[2])
        if y_pos < len(self.llane_x):
            centerroad = (self.llane_x[y_pos] + self.rlane_x[y_pos]) / 2
            centercam = undist_img.shape[1] / 2
            self.veh_pos = round((centerroad - centercam) * self.x_m_per_pix, 2)

    def draw_result(self, undist_img, warped_img, m_inv):
        """Draw the end result on a frame"""
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped_img).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        if len(self.llane_x) > 1:
            pts_left = np.array([np.transpose(np.vstack([self.llane_x, self.plot_y]))])
            pts_right = np.array(
                [np.flipud(np.transpose(np.vstack([self.rlane_x, self.plot_y])))])
            pts = np.hstack((pts_left, pts_right))
            # pts = np.array(pts, dtype=np.int32)
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            lane_curve = round((self.lrad_curve + self.rrad_curve) / 2, 2)
            # Warp the blank back to original image space using inverse perspective
            # matrix (Minv)
            unwarp = cv2.warpPerspective(
                color_warp, m_inv, (undist_img.shape[1], undist_img.shape[0]))
            # Combine the result with the original image
            result = cv2.addWeighted(undist_img, 1, unwarp, 0.3, 0)
            cv2.putText(result, "Lane curve: {} m".format(lane_curve), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 255, 255), thickness=2)
            if self.veh_pos > 0:
                pos_text = "Vehicle pos.: {} m (left from center lane) ".format(
                    abs(self.veh_pos))
            else:
                pos_text = "Vehicle pos.: {} m (right from center lane)".format(
                    abs(self.veh_pos))
            cv2.putText(result, "{}".format(pos_text), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 255, 255), thickness=2)

            cv2.putText(result, "Detect left line: {}".format(self.left_detected),
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 255, 255), thickness=2)
            cv2.putText(result, "Detect right line: {}".format(self.right_detected),
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 255, 255), thickness=2)

            if DEBUG:
                plt.imshow(result)
                plt.show()
            return result
        else:
            return undist_img