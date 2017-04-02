## Writeup Template


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This project needs these Python files:
1. *main.py*
2. *hog_classifier.py*
3. *vehiclefinder.py*
4. *car.py*
5. *lesson_functions.py*

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in *hog_classifier.py*. The file represents the HOG classifier class which contains constructor and several functions. The class is created and called in *main.py*.

The constructor (`__init__()`) contains all the HOG parameters. The function `init_features` has to be called first in order to reading all the `vehicle` and `non-vehicle` images. Using the `glob` library, I can acquire the datasets and extract their HOG features. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

These are the first two lines of codes that have to be called first:

    self.hog_clf = HogClassifier("HSV")
    self.hog_clf.init_features()

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I set the parameters as:

    self.orient = 8
    self.spatial = 32
    self.histbins = 32
    self.cell_per_block = 2
    self.pix_per_cell = 16
    self.hog_channel = "ALL"
    self.spatial_feat = True
    self.hist_feat = True
    self.hog_feat = True

With these parameters first I chose the color space YCrCb. However, after my experiments the color space HSV can better detect the white car. The extracted features contains  spatial feature, the color histogram, and three channel of the HOG features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM by calling the function `learn_SVC_features()`. I use 20% of the feature datasets as the test datasets.

The test accuracy of the SVC is 99.1 %.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window function can be found in the file *lesson_functions.py*. This file contains support functions which has been discussed in the lecture. In the class VehicleFinder, I define the window scales and the overlap scales:

    win_scales = [64, 96, 128]
    overlap = [0.75, 0.8, 0.8]


![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

So the overal pipeline is implemented in the `run()` function of the `VehicleFinder` class.

After defining the sliding windows, I pass the windows to the classfier in order to get the prediction and the confidence score. The confidence score helps filtering the false positive.

1. Create an empty list to receive positive detection windows
2. Iterate over all windows in the list
3. Extract the test window from original image
4. Extract features for that window using single_img_features()
5. Scale extracted features to be fed to classifier
6. Predict using your classifier
7. If positive (prediction == 1 and confidence score is higer than 0.99) then save the window
8. Return windows for positive detections

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

