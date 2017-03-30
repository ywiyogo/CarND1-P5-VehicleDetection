"""
Author: Y.Wiyogo
"""
import numpy as np
import cv2

from helper_methods import visualize_img
from lesson_functions import slide_window, draw_boxes
from hog_classifier import HogClassifier

DEBUG = 1


class VehicleFinder:
    """
    Vehicle Finder Class
    1. Run the sliding windows to get image boxes from the input image
    2. For each image box, pass this box to the classifier
    3. For each correct prediction, create a bounding box on it
    4. Returns the input image with the bounding boxes
    """

    def __init__(self):
        """Constructor"""
        self.window_list = None
        self.hog_clf = None

    def init_detection_algorithm(self, algorithm):
        """Initialize the object detection algorithm"""
        if algorithm.lower() == "hog":
            print("Initializing HOG classifier ...")
            self.hog_clf = HogClassifier()
            self.hog_clf.init_features()
            self.hog_clf.learn_SVC_features()

    def run(self, img):
        """Run Vehicle Finder Pipeline"""
        # Start top y: 720 -(32*11) = 368
        y_top = 368
        win_scales = [64, 96, 128]
        print(img.shape)

        windowslist = slide_window(img, [0, img.shape[1]], [y_top, img.shape[0]],
            (64, 64))
        visualize_img([draw_boxes(img, windowslist)], ["Test"], [0])

        classifier = self.hog_clf.svc
        swindows = self.search_windows(img, windowslist, classifier, self.hog_clf.scaler_feat)
        res_imgs = [draw_boxes(img, swindows)]
        visualize_img(res_imgs, ["Test"], [0])

    def detect(self, img):
        """Run detection"""

        # self.hog_features = self.extract_features()
        # self.visualize_hog_feature(img)

    def search_windows(self, img, windows, clf, scaler):
        """Define a function you will pass an image
        and the list of windows to be searched (output of slide_windows())
        """
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
        # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
            features = self.hog_clf.single_img_features(test_img)
            # print("Shape: ", features.shape)
            # print("Scaler Shape: ", scaler.scale_.shape)
        # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(features.reshape(1, -1))
        # 6) Predict using your classifier
            prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows
