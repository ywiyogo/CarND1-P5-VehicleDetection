"""
Author: YWiyogo
"""
import matplotlib.image as mpimg
#from moviepy.editor import VideoFileClip
from calibration import Calibration
from vehiclefinder import VehicleFinder


DEBUG = 0
calibrator = None
hog_classifier = None
veh_finder = None


def process_image(rgb_img):
    """
    NOTE: The output you return should be a color image (3 channel) for
    processing video below
    """
    global calibrator, veh_finder
    # Compute the camera calibration matrix and distortion coefficients given
    if calibrator is None:
        calibrator = Calibration()
    undist_img = calibrator.undistort_img(rgb_img)


# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.


def main():
    """Main function"""
    # 1. Perform a HOG feature extraction on a labeled training set of images
    # and train a classifier Linear SVM classifier

    veh_finder = VehicleFinder()
    veh_finder.init_detection_algorithm("HOG")
    # Get image from video
    rgb_img = mpimg.imread("./test_images/test1.jpg")
    # process_image(rgb_img)
    # extract feature
    veh_finder.run(rgb_img)



if __name__ == '__main__':
    main()