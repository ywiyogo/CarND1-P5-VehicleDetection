"""
Author: YWiyogo
Calibration class
"""
import cv2
import numpy as np
import glob
import pickle
from os.path import isfile
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

DEBUG = 0


class Calibration:
    """Calibration class
    The dimension of the chessboard is 6x9
    Even though the chessboard can be 2D, the object points has to be
    3 dimensional due to the camera matrix for the funt cv2.undistort
    cv2.error: (-210) objectPoints should contain vector of vectors of points of
    type Point3f in function collectCalibrationData
    """

    def __init__(self):
        """Contructor"""
        self.mtx = []       # camera matrix
        self.dist = []      # distortion coefficients
        self.rvecs = []     # rotation vector
        self.tvecs = []     # translation vector
        opoints = np.zeros((9 * 6, 3), np.float32)
        opoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        # check for an existing data
        fname = "calibration_data.p"
        if isfile(fname):
            print("Calibration data found")
            with open(fname, 'rb') as f:
                calib_data = pickle.load(f)
                self.mtx = calib_data["mtx"]
                self.dist = calib_data["dist"]
                self.rvecs = calib_data["rvecs"]
                self.tvecs = calib_data["tvecs"]
        else:
            # iterate all the chessboard images
            cal_files = glob.glob('./camera_cal/calibration*.jpg')
            for f in cal_files:
                if DEBUG: print("Filename: ", f)
                img = cv2.imread(f)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                if ret is True:
                    objpoints.append(opoints)
                    imgpoints.append(corners)
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    # gray.shape[::-1] invert the shape:(720, 1280)->(1280, 720)
                    ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
                        cv2.calibrateCamera(objpoints, imgpoints,
                                            gray.shape[::-1], None, None)
                else:
                    print("Warning: findChessboardCorners failed")
            calib_data = {}
            calib_data["mtx"] = self.mtx
            calib_data["dist"] = self.dist
            calib_data["rvecs"] = self.rvecs
            calib_data["tvecs"] = self.tvecs
            pickle.dump(calib_data, open(fname, "wb"))

    def undistort_img(self, img):
        """Undistorting image function"""
        if DEBUG: print(self.mtx)
        undist_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        if DEBUG:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image')
            ax2.imshow(undist_img)
            ax2.set_title('Undistorted Image')
            plt.show()
        return undist_img
