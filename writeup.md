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
[image1]: ./examples/
[image2]: ./examples/
[image3]: ./examples/
[image4]: ./examples/
[image5]: ./examples/
[image6]: ./examples/
[image7]: ./examples/
[video1]: ./project_video_result.mp4

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


|Test | setse |
|--|--|
|arwaf| * etsete 
        * seftsef|
        * esse  |

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

First, I tried the RGB color channel for the feature extraction. I combine all the features from color feature spatial, color histogram and the HOG features. My start parameter values was defined as these:

* HOG orientation: 8
* HOG pixel/cell: 16
* HOG cell/block: 2
* Binned color feature spatial: 16
* Color histogram bins: 16

After run the C-Support Vector classifier, it returns 98.23% for the test accuracy. From this result, I changed only the color space and observing the results in order to get the best color space. The table below shows my experiment result with all datasets provided by Udacity (GTI and KITTI).

|Color|Orient|Spatial|Hist Bins|pix per cell|SVC vec.length|Accuracy|
|--|--|--|--|--|--|--|--|
|RGB|8|16|16|16|1680|98.23%|
|HSV|8|16|16|16|1680|99.16%|
|YCrCb|8|16|16|16|1680|98.99%|
|YUV|8|16|16|16|1680|99.01%|

The comparison places HSV and YUV as my favorite color spaces to be used, compared to RGB. I continued my experiment by tuning the other parameter values, and get these comparisons:

|Color|Orient|Spatial|Hist Bins|pix per cell|SVC vec.length|Accuracy|
|--|--|--|--|--|--|--|--|
|YUV|8|16|16|16|1680|99.01%|
|YUV|8|16|16|8|5520|99.32%|
|YUV|12|16|16|8|7872|99.47%|
|YUV|8|32|32|8|7872|99.55%|
|YUV|8|32|32|16|4032|99.13%|
|HSV|8|32|32|16|4032|99.21%|
|YUV|12|32|32|16|4464|99.52%|

Based on the above table, HSV has the based accuracy. However, after I tried with the real test images. The feature extraction using HSV returns more false positive on the yellow lane rather than using YUV color space. Because of this reason I change my decision to set the color space to YUV.
Finally, these are the fixed parameters that I use as:

    def __init__(self, cspace="YUV"):
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

The sliding window function can be found in the file *lesson_functions.py*. This file contains others support functions which has been discussed in the lecture. 
In the class VehicleFinder, I define the window scales and the overlap scales:

    win_scales = [64, 96, 128]
    overlap = [0.75, 0.8, 0.8]

Windows size more than 128 x 128 pixels returns a more broaded detection on the heatmap. As the result, we will often get a larger bounding box than the object size. 

Using overlap between 0.7 and 0.8 creates a higher value of heatmap. This approach results more stability of the resulting bounding box.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

So the overal pipeline is implemented in the `run()` function of the `VehicleFinder` class.

After defining the sliding windows, I pass the windows to the classfier in order to get the prediction and the confidence score. Here are several optimization that I have done:

1. utilize the confidence score function of the SVC and filter the score
2. applying the previous bounding box from the last frame to the heatmap
3. Apply a heatmap threshold of 1
4. Create a car object for each vehicle detection

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

