## Writeup

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
[image1]: ./examples/car_noncar.png
[image2]: ./examples/car_hog_imgs.png
[image3]: ./examples/noncar_hogimgs.png
[image3a]: ./examples/scaledfeatures.png
[image4]: ./examples/slidingwindows.png
[image5]: ./examples/slidingwindows2.png
[image6]: ./examples/swindows_heatmap.png
[image7]: ./examples/swindows_heatmap2.png
[image8]: ./examples/
[image7]: ./examples/
[video1]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Code Structure

This project needs these Python files:
1. *main.py*
2. *hog_classifier.py*
3. *vehiclefinder.py*
4. *car.py*
5. *lesson_functions.py*

Additionally, I added the code from P4 together:
1. *calibration_data.p*
2. *lanefinder.py*
3. *perspective_transform.py*

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First, I define the HOG parameters in the constructor of my HOG classifier class (can be found in *hog_classifier.py*). Using `glob` library I can extract all of the given datasets.

I combine the all the feature extraction methods (binned spatial, color histogram and the HOG) in the class HogClassifier. The extraction is quite similar to the lesson functions which uses the function from `skimage.feature`. This library provides also hog_image as the second return values if I set the `vis` argument  to `True`.

The implementation can be found in the *hog_classifier.py* inside the function`extract_features()` and *lesson_functions.py* inside the function `get_hog_features`

The function `init_features` has to be called first in order to reading all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` image:

![alt text][image1]


These are the first two lines of codes that have to be called first in the class VehicleFinder:

    HogClassifier()
    init_features()


#### 2. Explain how you settled on your final choice of HOG parameters.

First, I tried the RGB color channel for the feature extraction. I combine all the features from color feature spatial, color histogram and the HOG features. My start parameters were defined as these:

* HOG orientation: 8
* HOG pixel/cell: 16
* HOG cell/block: 2
* Binned color feature spatial: 16
* Color histogram bins: 16

After run the C-Support Vector classifier, it returns 98.23% for the test accuracy. From this result, I changed only the color space and observing the results in order to get the best color space. The table below shows my experiment result with all datasets provided by Udacity (GTI and KITTI). The values in the SCV vector length column are generated from the `extracted_features()` function. The last column presents the accuracy value using 20% of the total datasets as test datasets.

| Color | Orient | Spatial | Hist Bins | Pix per cell | SVC vec.length | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|RGB|8|16|16|16|1680|98.23%|
| <div style="background-color:yellow;">HSV</div> |8|16|16|16|1680|<div style="background-color:yellow;">99.16%</div>|
|YCrCb|8|16|16|16|1680|98.99%|
| <div style="background-color:yellow;">YUV</div> |8|16|16|16|1680|<div style="background-color:yellow;">99.01%</div>|

The comparison places HSV and YUV as my favorite color spaces to be used, compared to RGB. I continued my experiment by tuning the other parameter values, and get these comparisons:

|Color|Orient|Spatial|Hist Bins|pix per cell|SVC vec.length|Accuracy|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|YUV|8|16|16|16|1680|99.01%|
|YUV|8|16|16|8|5520|99.32%|
|YUV|12|16|16|8|7872|99.47%|
|YUV|8|32|32|8|7872|99.55%|
|YUV|8|32|32|16|4032|99.13%|
|HSV|8|32|32|16|4032|99.21%|
|YUV|12|32|32|16|4464|99.52%|

Based on the above table, the pixel per cell 8 returns the higher accuracy. However, it consumes a lot of memory (with vector length 7872). Spatial size 32 returns a better accuracy than 16, since decompressing the original image from 64x64 to 16x16 will lose a lot of information of images.

HSV has the best accuracy, but after I tried with the real test images, it returns more false positive on the yellow lane rather than using YUV color space. Because of this reason I change my decision to set the color space to YUV. The HOG features of the YUV color space can be visualized as this images below (`orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`):

| Car | Not car |
|:--:|:--:|
|![alt text][image2]|![alt text][image3]|

Finally, I set the parameters as:

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


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM by calling the function `learn_SVC_features()`. I use 20% of the feature datasets as the test datasets. The function can be found in the HogClassifier class. Before I train the extracted features, I need to normalized them with `StandardScaler` from sklearn library. The below figure presents the visualisation of a normalized features of a car:

![alt text][image3a]

Using the above HOG parameters, I get the test accuracy of the SVC is 99.13 %.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window function can be found in the file *lesson_functions.py* as:

    def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))

The file also contains others support functions which has been discussed in the lessons. 
The `slide_window()` function is called in the class VehicleFinder. Before that, I define the window scales, the y-axis limit and the overlap scales as:

    win_scales = [64, 96, 128]
    ybottom = [600, 680, img.shape[0]]
    overlap = 0.8

I chose 64 because our datasets has 64x64 pixel. Windows size more than 128 x 128 pixels returns a more broader detection. As the result, we will often get a larger bounding box than the object size. Furthermore, the vertical start on the y-axis is **shifted by 20 pixels** for each iteration, since larger objects have higher y-axis position:

    for i, scl in enumerate(win_scales):
        windowslist = slide_window(img, [0, img.shape[1]], [y_top+i*20, ybottom[i]], (scl, scl), (overlap, overlap))

Based on my experiments, overlap value of 0.8 provides more positive detection windows. This approach results more stability of the resulting bounding box. The below figure shows the sliding windows approach using my SVC classifier. The left column shows the area of the sliding windows

##### Example 1
![alt text][image4]

##### Example2
![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

So the overal pipeline is implemented in the `run()` function of the `VehicleFinder` class.

After defining the sliding windows, I pass the windows to the classfier in order to get the prediction and the confidence score. Here are several optimization that I have done:

1. utilize the confidence score function `decision_function()` of the SVC and filter the score
2. applying the previous bounding box from the last frame to the heatmap
3. Instead of thresholding the heatmap value, I calculate the maximum value of a detected labels (cropped heatmap area), see the function `evaluate_labels(self, labels)` in *vehiclefinder.py*.
4. Filter the area based on the maximum heatmap. I set the threshold value to 2, means that a label which has maximum 2 heatmap will be ignored (*vehiclefinder.py* line 149).
4. Create a car object for each vehicle detection

The below figures visualizes the conversion from the step 1 and step 2 from my two images above. In example 1 the heatmap region with maximum value of 1 will be discarded:

##### Example 1
![alt text][image6]

##### Example 2
![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here I provide two videos:
1. a [link to my video result without tracking](./project_video_result-t2-notrack.mp4) and 
2. a [link to my video result with tracking](./project_video_result-t2-track.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions, in my case the threshold is 2. This is implemented in the function:

    def evaluate_labels(labels)

the function is called in the `run()` function (line 105 in *vehiclefinder.py*).

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Since using the heatmap threshold does not avoid 100% of false positive, I implemented a simple tracking functionality. For this purpose, I create a Car class in *car.py*. 

The function `def update_state(self, new_bbox)` will update the internal state of a car object. The centroid of the given bounding box will be calculated and compared to the previous bounding box. If the centroid is located inside the previous bounding box or the distance to the previous centroid is less than a define threshold, then the new bounding box is assign to this car object.  The logic implementation can be found in the function `check_centroid()` (line 45-54). 

In my second video, we can see that a new car ID is created because in a one frame the centroid of the new detection is more than the threshold. 

Additionally, the function `is_valid()` provides a validity check of a car object whether the car can be removed or not. The tradeof of this tracking function is the calculation time per frame. By activating the tracking mode, it needs 2-3 seconds more to perform the algorithm per frame.

