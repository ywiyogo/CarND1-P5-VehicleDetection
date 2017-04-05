"""
Author: Y.Wiyogo
"""
import numpy as np
import cv2
from helper_methods import visualize_imgs, two_cols_vis_imgs
from lesson_functions import slide_window, find_cars, add_heat
from lesson_functions import apply_threshold, draw_labeled_bboxes, draw_boxes
from hog_classifier import HogClassifier
from scipy.ndimage.measurements import label
from car import Car
import matplotlib.pyplot as plt
DEBUG = 1


class VehicleFinder:
    """
    Vehicle Finder Class
    1. Run the sliding windows to get image boxes from the input image
    2. For each image box, pass this box to the classifier
    3. For each correct prediction, create a bounding box on it
    4. Returns the input image with the bounding boxes
    """

    def __init__(self, track=False, threshold=2):
        """Constructor"""
        self.heatmap = None
        self.hog_clf = None
        self.cars = {}
        self.bboxs = []  # list of bounding box
        self.tracking = track
        self.heatmap_thres = threshold
        print("Tracking: ", self.tracking, " Heatmap threshold: ",
              self.heatmap_thres)

    def init_detection_algorithm(self, algorithm):
        """Initialize the object detection algorithm"""
        if algorithm.lower() == "hog":
            print("Initializing HOG classifier ...")
            self.hog_clf = HogClassifier()
            self.hog_clf.init_features()
            self.hog_clf.learn_SVC_features()

    def run(self, img, debug=False):
        """Run Vehicle Finder Pipeline"""
        global DEBUG
        DEBUG = debug
        # Start top y
        y_top = 380

        for car in list(self.cars.values()):
            car.frame_update = False
        if DEBUG:
            print("Input img: ", img.shape)

        win_scales = [64, 96, 128]
        ybottom = [600, 680, img.shape[0]]
        overlap = 0.8
        allwindows = []
        if DEBUG:
            imgs1 = {}
            imgs2 = {}
            imgcopy1 = np.copy(img)
            imgcopy2 = np.copy(img)
        #-----------------------------------
        # Start sliding windows
        for i, scl in enumerate(win_scales):
            windowslist = slide_window(img, [0, img.shape[1]], [y_top+i*20, ybottom[i]],
                                       (scl, scl), (overlap, overlap))
            classifier = self.hog_clf.svc
            swindows, scores = (self.search_windows(img, windowslist, classifier, self.hog_clf.scaler_feat))
            allwindows = swindows + allwindows

            if DEBUG:
                imgs1[scl] = draw_boxes(imgcopy1, windowslist)
                imgs2[scl] = draw_boxes(imgcopy2, swindows)

        if DEBUG:
            two_cols_vis_imgs(imgs1, imgs2)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, allwindows)
        #heat = apply_threshold(heat, 0)

        # Visualize the heatmap when displaying
        self.heatmap = np.clip(heat, 0, 255)

        if DEBUG:
            allwin_img = draw_boxes(np.copy(img), allwindows)
            visualize_imgs([allwin_img, self.heatmap], ["Combined Windows", "Heatmap"], [None, "hot"])
        # Find final boxes from heatmap using label function
        labels = label(self.heatmap)
        self.evaluate_labels(labels)
        #print(len(labels[0]), "---", labels[1])
        #res_img = draw_labeled_bboxes(np.copy(img), labels, heatmap, vis=debug)

        # Filtering cars which is not up-to-date
        new_car_dict = {}
        for car in list(self.cars.values()):
            # print("carID: ", car.carID, " ", car.tracked_count, " ", car.frame_update_count)
            if car.is_valid():
                new_car_dict[car.carID] = car
        self.cars = new_car_dict

    def draw_cars(self, img):
        """Draw all car bbox on the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        for car in list(self.cars.values()):
            if self.tracking:
                if car.tracked_count > 0:
                    cv2.rectangle(img, car.bbox[0], car.bbox[1], (0, 0, 255), 6)
                    cv2.putText(img, str("car " + str(car.carID)),
                                ((car.bbox[0][0], car.bbox[0][1]-5)), font, 1,
                                (255, 255, 255), 2)
            else:
                cv2.rectangle(img, car.bbox[0], car.bbox[1], (0, 0, 255), 6)
                cv2.putText(img, str("car "),
                            ((car.bbox[0][0], car.bbox[0][1]-5)), font, 1,
                            (255, 255, 255), 2)
        if DEBUG:
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(self.heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()
        return img

    def evaluate_labels(self, labels):
        pos_bbox = []
        if not self.tracking: self.cars = {}

        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            subheatmap = self.heatmap[bbox[0][1]:bbox[1][1],
                                      bbox[0][0]:bbox[1][0]]
            flatmap = subheatmap.flatten()
            # threshold for heatmap max
            #print("max subheatmap: ", max(flatmap))
            if max(flatmap) > self.heatmap_thres:
                pos_bbox.append(bbox)
                if self.tracking:
                    self.update_cars(bbox)
                else:
                    #print("Create a new car: ", car_number)
                    newcar = Car(car_number, bbox)
                    self.cars[car_number] = newcar

        self.bboxs = pos_bbox

    def update_cars(self, bbox):
        found = False
        for car in list(self.cars.values()):
            if car.check_bbox(bbox):
                car.update_state(bbox)
                found = True
                break
        if not found:
            car_id = 0
            for i in range(1, len(self.cars) + 2):
                if not (i in self.cars.keys()):
                    car_id = i
                    break
            print("Create a new car: ", car_id)
            newcar = Car(car_id, bbox)
            self.cars[car_id] = newcar

    def filter_labels(self, labels):
        """Filter the generated label from heatmap"""
        # self.labels[1] is the object number
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Ignore very narrow heat regions that less than  pixels

            width = np.max(nonzerox) - np.min(nonzerox)
            minwidth = 45
            minwidth1 = 55
            # and ignore small box for near objects (y-axis > 500)
            det = False if width < minwidth1 and np.max(nonzeroy) > 500 else True
            if width > minwidth and det:
                found = False
                for car in list(self.cars.values()):
                    if car.check_bbox(bbox):
                        car.update_state(bbox)
                        found = True
                        break
                if not found:
                    car_id = 0
                    for i in range(1, len(self.cars) + 2):
                        if not (i in self.cars.keys()):
                            car_id = i
                            break
                    print("Create a new car: ", car_id)
                    newcar = Car(car_id, bbox)
                    self.cars[car_id] = newcar

    def search_windows(self, img, windows, clf, scaler):
        """Define a function you will pass an image
        and the list of windows to be searched (output of slide_windows())
        """
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        conf_score = []
        # 2) Iterate over all windows in the list
        for window in windows:
        # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
            features = self.hog_clf.single_img_features(test_img)
        # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(features.reshape(1, -1))
        # 6) Predict using your classifier
            prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
        # 8) Calculate the SVC confidence score
                c_score = clf.decision_function(test_features)
                if c_score > 0.7:
                    on_windows.append(window)
                    conf_score.append(c_score)

        # 8) Return windows for positive detections
        return on_windows, conf_score
