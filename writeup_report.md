# **Behavioral Cloning** 

## Writeup Udacity-P3 by Zhong

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/CNN_architecture_by_NAVIDA.png "CNN_architecture_by_NAVIDA"
[image2]: ./examples/center_2018_07_21_20_36_36_108.jpg "center_2018_07_21_20_36_36_108"
[image3]: ./examples/center_2018_07_21_20_39_00_838.jpg "Recovery Image"
[image4]: ./examples/center_2018_07_21_20_39_01_847.jpg "Recovery Image"
[image5]: ./examples/center_2018_07_21_20_39_02_264.jpg "Recovery Image"
[image6]: ./examples/center-2017-02-06-16-20-04-855.jpg "Normal Image"
[image7]: ./examples/center-2017-02-06-16-20-04-855-flipped.jpg "Flipped Image"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md for summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model use the architecture published by the autonomous vehicle team at NVIDIA. The model consists 5 Convolutional layer and 3 Fully-connected layer(model.py lines 48-61). You can see the architecture below.

The model is normalized in the model using a Keras lambda layer (code line 49), and cut the images (up 70 rows pixels and down 25 rows pixels)using a Keras Cropping2D layer (code line 50)


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 56). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
Three laps of center lane driving 
One lap of recovery driving from the sides
One lap of focusing on driving smoothly around curves
One lap of opposite direction


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduse mean squared error on training set and validation set.

My first step was to use a convolution neural network model similar to the Lenet I thought this model might be appropriate because the Lenet is frequently-used. But I found it is work well for the test even have lower squared error on training set and validation set.

Then I use the architecture published by the autonomous vehicle team at NVIDIA. The model consists 5 Convolutional layer and 3 Fully-connected layer(model.py lines 48-61). 

When I set the np_epoch 5 , I got a low mean squared error on training set but a high mean squared error on validation set at last. So I add a Dropout layer and set the np_epoch to 4 to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track in turning or in the bridge, to improve the driving behavior in these cases, I use my own collected data for tarining.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 48-61) consists 5 Convolutional layer and 3 Fully-connected layer.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the sides. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded one lap of opposite direction

To augment the data sat, I also flipped images and angles thinking that this would balance the traing set have too much left-turn. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
