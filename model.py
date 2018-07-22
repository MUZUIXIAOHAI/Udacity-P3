import csv
import cv2
import numpy as np

#read the driving_log.csv. my data put in the folder 'data_1'.
lines = []
with open('./data_1/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for i,line in enumerate(reader):
    		if i > 0:
        		lines.append(line)

#upload the data from 'data_1' folder, include center images ,right images and left images.
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data_1/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    measurement = float(line[3])
    #add or subtract adjusted steering measurements for the side camera images
    measurements.append(measurement)
    measurements.append(measurement+0.2)#left images
    measurements.append(measurement-0.2)#right images
    
#augment the images by flipping the images and measurement * -1.0
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

#train set
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#import keras library function
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#define my model , here I use is the architecture published by the autonomous vehicle team at NVIDIA
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(36, (5, 5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(48, (5, 5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#train the model
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 4, verbose = 1)

#save the model as model.h5
model.save('model.h5')

#here is for visualizing loss
# print(history_object.history.keys())

# import matplotlib.pyplot as plt

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc = 'upper right')
# plt.show()


