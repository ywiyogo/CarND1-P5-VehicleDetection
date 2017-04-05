# Vehicle Detection

Udacity - Self-Driving Car NanoDegree 5th project about vehicle detection.

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## About this repository

This repository is a project that shows how vehicle detection based on the computer vision algorithms works. The algorithm works offline and not in real-time.

Note: this repository contains also the solutions from the [fourth project](https://github.com/ywiyogo/CarND1-P4-AdvancedLaneFinding) to visualize the lane detection.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.

## Requirements

Install the Udacity starting kit from <https://github.com/udacity/CarND-Term1-Starter-Kit>

## Installation

    git clone https://github.com/ywiyogo/CarND1-P5-VehicleDetection.git

## Usage

1. Go to the repository folder and activate the virtualenvironment:

        source activate carnd-term1

2. Start the program 
        python main.py

## Issues

[Issues](https://github.com/ywiyogo/CarND1-P5-VehicleDetection/issues)

## Roadmap

[Milestones](https://github.com/ywiyogo/CarND1-P5-VehicleDetection/milestones)

