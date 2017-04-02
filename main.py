"""
Author: YWiyogo
"""
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from calibration import Calibration

#to cut video
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# -- P4 --
from perspective_transform import PerspectiveTransform
from lanefinder import LaneFinder
from helper_methods import sobel_color_threshold
#----

# --- P5 -----
from vehiclefinder import VehicleFinder
# ------


DEBUG = 0
calibrator = None
transformer = None
lanefinder = None

hog_classifier = None
veh_finder = None


def process_image(rgb_img):
    """
    NOTE: The output you return should be a color image (3 channel) for
    processing video below
    """
    # --------------P4 Lane Detection ----------------------------
    global calibrator
    global transformer
    global lanefinder
    # Compute the camera calibration matrix and distortion coefficients given
    if calibrator is None:
        calibrator = Calibration()
    undist_img = calibrator.undistort_img(rgb_img)

    #  Use color transforms, etc.,to create a thresholded binary image.
    # -----------------------------------------------------------------
    thres_img = sobel_color_threshold(undist_img)
    if transformer is None:
        transformer = PerspectiveTransform()

    #  Apply a perspective transform to rectify binary image ("birds-eye view").
    transformer.do_perpectivetransform()
    warped_img = transformer.warp(thres_img)

    if lanefinder is None:
        lanefinder = LaneFinder()
    # Perform the lane detection
    leftx, rightx = lanefinder.detect_lanes(warped_img)
    lanefinder.calc_radcurvature()
    lanefinder.calc_vehicle_position(undist_img)
    lane_img = lanefinder.draw_result(undist_img, warped_img, transformer.Minv)
    # --------------P5 Vehicle Detection ----------------------------
    global veh_finder
    veh_finder.run(undist_img, debug=False)
    # draw cars with lane
    res_img = veh_finder.draw_cars(lane_img)
    return res_img


# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.


def main():
    """Main function"""
    # 1. Perform a HOG feature extraction on a labeled training set of images
    # and train a classifier Linear SVM classifier
    global veh_finder
    veh_finder = VehicleFinder()
    veh_finder.init_detection_algorithm("HOG")
    #ffmpeg_extract_subclip("project_video.mp4", 37, 44, targetname="test2.mp4")
    if 1:
        # vidfilename = "test2.mp4"
        # write_output = "test2_result.mp4"
        vidfilename = "project_video.mp4"
        write_output = "project_video_result.mp4"
        clip = VideoFileClip(vidfilename)
        proc_clip = clip.fl_image(process_image)
        proc_clip.write_videofile(write_output, audio=False)
    else:
        # Get image from video
        file = "./test_images/test"
        for i in range(2,9):
            filename = file+str(i)+".jpg"
            rgb_img = mpimg.imread(filename)
            veh_finder.run(rgb_img, debug=True)
            res_img = veh_finder.draw_cars(rgb_img)
if __name__ == '__main__':
    main()
