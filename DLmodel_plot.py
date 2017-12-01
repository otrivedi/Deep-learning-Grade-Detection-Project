#!/bin/env python3
#SBATCH -N 1 # No. of computers you wanna use. Typically 1
#SBATCH -n 2 # No. of CPU cores you wanna use. Typically 1
#SBATCH -p gpu # This flag specifies that you wanna use GPU and not CPU
#SBATCH -o roadseg.out # output file name, in case your program has anything to output (like print, etc)
#SBATCH -t 24:00:00 # Amount of time
#SBATCH --gres=gpu:2 # No. of GPU cores you wanna use. Usually 2-3
import numpy
import pandas
import cv2
import math
import os
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

K = []

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataY = []
	for i in range(len(dataset)-look_back-1):
		dataY.append(dataset[i + look_back, :])
	return numpy.array(dataY)

def myGenerator():
	# load the dataset
	dataframe = pandas.read_csv('roadseg_dataset_newest.csv', header=1)
	dataset = dataframe.values
	data = dataset[10000:50000,:]

	for i in data[:,0]:
		img = cv2.imread(i)
		img = cv2.resize(img, (100,100))
		K.append(img)
	X = numpy.array(K)

	look_back = 30
	Y = numpy.array(data[:,2:4])
	# split into train and test sets
	train_size = int(len(data) * 0.67)
	test_size = len(data) - train_size
	X_train = X[0:(train_size)-look_back-1]
	X_test = X[(train_size):len(X)-look_back-1]
	#train = data[0:train_size]
	#test = data[train_size:len(data)]
	Y_train = Y[0:train_size,:]
	Y_test = Y[train_size:len(dataset),:]
	# reshape into X=t and Y=t+1
	Y_train = create_dataset(Y_train, look_back)
	Y_test = create_dataset(Y_test, look_back)
	X_train = X_train.reshape(X_train.shape[0], 3, 100, 100).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], 3, 100, 100).astype('float32')
	X_train = X_train / 255
	X_test = X_test / 255
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
	while 1:
		for i in range(1250): # 1250 * 32 = 40000 -> # of training samples
			if i%125==0:
				print("i = " + str(i))
			yield X_train[i*32:(i+1)*32], Y_train[i*32:(i+1)*32]

batch_size = 128
nb_classes = 10
nb_epoch = 12
num_classes = 2
def model():
	# create model
	model = Sequential()
	model.add(Conv2D(32,3, 3, border_mode='same', input_shape=(3, 100, 100), activation='relu'))
	model.add(Dropout(0.15))
	model.add(Conv2D(32,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64,3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.15))
	model.add(Conv2D(64,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128,3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.15))
	model.add(Conv2D(128,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.15))
	model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.15))
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.15))
	model.add(Dense(num_classes))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = model()
# Fit the model
model.fit_generator(myGenerator(), samples_per_epoch = 40000, nb_epoch = 2, verbose=2, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=10, batch_size=150, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))

# Estimate model performance
trainScore = model.evaluate(X_train, Y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
# 	json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# #print("Saved model to disk")
