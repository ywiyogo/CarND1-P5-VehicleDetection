"""
HOG Classifier
Author: YWiyogo
"""
import glob
import pickle
import numpy as np
import cv2
import time
from os.path import isfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from lesson_functions import color_hist, bin_spatial, get_hog_features
DEBUG = 1


class HogClassifier:
    """HOG Classifier Class"""

    def __init__(self):
        """Constructor"""
        self.cars_features = None
        self.noncars_features = None
        self.scaled_features = None
        self.scaler_feat = None
        self.svc = None
        self.cars_data = '../p5_vehicles/**/*.png'
        self.noncars_data = '../p5_non-vehicles/**/*.png'
        self.orient = 8
        self.spatial = 32
        self.histbins = 32
        self.hog_channel = 0
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True
        self.color_space = 'RGB'
        self.spatial_size = (32, 32)

    def init_features(self):
        """Initialize hog features by loading the labeld images"""
        fname = "data_features.p"
        scaled_fname = "data_scaler.p"

        if isfile(fname):
            print("features data found")
            with open(fname, 'rb') as f:
                feature_data = pickle.load(f)
                self.cars_features = feature_data["cars"]
                self.noncars_features = feature_data["noncars"]

        else:
            car_images = glob.glob(self.cars_data, recursive=True)
            noncar_images = glob.glob(self.noncars_data, recursive=True)
            print("total car imgs: {}".format(len(car_images)))
            print("total not car img: {}".format(len(noncar_images)))
            self.cars_features = self.extract_features(car_images)
            # print("Extracted car features {}".format(self.cars_features))
            self.noncars_features = self.extract_features(noncar_images)
            # print("Extracted noncar features {}".format(self.noncars_features))
            feature_data = {}
            feature_data["cars"] = self.cars_features
            feature_data["noncars"] = self.noncars_features
            pickle.dump(feature_data, open(fname, "wb"))
            print("Dumping features to {}.....".format(fname))
            # Create an array stack of feature vectors
            # feature_list = [self.cars_features, self.noncars_features]
            # Create an array stack, NOTE: StandardScaler() expects np.float64

        print("create vstack")

        if isfile(scaled_fname):
            print("scaled features data found")
            with open(scaled_fname, 'rb') as f:
                scaler_feature_data = pickle.load(f)
                self.scaled_features = scaler_feature_data["scaled"]
                self.scaler_feat = scaler_feature_data["scaler"]
        else:
            print("Length car: {}, ", len(self.cars_features))
            X = np.vstack((self.cars_features, self.noncars_features)).astype(
                np.float64)
            # Fit a per-column scaler
            self.scaler_feat = StandardScaler().fit(X)
            # Apply the scaler to X
            self.scaled_features = self.scaler_feat.transform(X)

            scaler_feature_data = {}
            scaler_feature_data["scaled"] = self.scaled_features
            scaler_feature_data["scaler"] = self.scaler_feat
            pickle.dump(scaler_feature_data, open(scaled_fname, "wb"))

    def learn_SVC_features(self):
        """Learning from the scaled features"""
        svc_file = "data_svc.p"
        if isfile(svc_file):
            print("SVC data found")
            with open(svc_file, 'rb') as f:
                self.svc = pickle.load(f)
        else:
            # Define the labels vector
            y = np.hstack((np.ones(len(self.cars_features)),
                           np.zeros(len(self.noncars_features))))
            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                self.scaled_features, y, test_size=0.2, random_state=rand_state)

            print('Using spatial binning of:', self.spatial, 'and',
                  self.histbins, 'histogram bins')
            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC
            self.svc = LinearSVC()
            # Check the training time for the SVC
            t = time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t = time.time()
            n_predict = 10
            print('My SVC predicts: ', self.svc.predict(X_test[0:n_predict]))
            print('For these', n_predict, 'labels: ', y_test[0:n_predict])
            t2 = time.time()
            print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
            # write to file
            pickle.dump(self.svc, open(svc_file, "wb"))

    def visualize_hog_feature(self, img, hog_features):
        """Visualization"""
        if len(hog_features) > 0:
            # Create an array stack of feature vectors
            vst_feature = np.vstack(hog_features).astype(
                np.float64)
            # Fit a per-column scaler
            vst_scaler = StandardScaler().fit(vst_feature)
            # Apply the scaler to X
            scaled_features = vst_scaler.transform(vst_feature)
            # Plot an example of raw and scaled features
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(img)
            plt.title('Original Image')
            plt.subplot(132)
            plt.plot(vst_feature)
            plt.title('Raw Features')
            plt.subplot(133)
            plt.plot(scaled_features)
            plt.title('Normalized Features')
            fig.tight_layout()
            plt.show()
        else:
            print('Your function only returns empty feature vectors...')

    def extract_features(self, img_files, pix_per_cell=8, cell_per_block=2):
        """
        Define a function to extract features from a list of images
        Have this function call bin_spatial() and color_hist()
        """
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in img_files:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if self.color_space != 'RGB':
                if self.color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif self.color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif self.color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif self.color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)

            if self.spatial_feat == True:
                spatial_features = bin_spatial(feature_image, size=self.spatial_size)
                file_features.append(spatial_features)
            if self.hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(feature_image, nbins=self.histbins)
                file_features.append(hist_features)
            if self.hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(get_hog_features(feature_image[:,:,channel],
                                            self.orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = get_hog_features(feature_image[:,:,self.hog_channel], self.orient,
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    def single_img_features(self, img, pix_per_cell=8, cell_per_block=2):
        """
        Define a function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images
        """
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)
        # 3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
        # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.histbins)
        # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(
                        feature_image[:, :, channel], self.orient, pix_per_cell,
                        cell_per_block, vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(
                    feature_image[:, :, self.hog_channel], self.orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)
