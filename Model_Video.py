import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import csv
import pandas
import math
from skvideo.io import VideoWriter

def slope(pitch_1,pitch_2,velocity,time):
    slope = numpy.around(numpy.rad2deg(numpy.arcsin(-(pitch_1-pitch_2)/(velocity*time))), decimals=2)
    return slope

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1

K = []
dataframe2 = pandas.read_csv('road_dataset_newest.csv')
dataset2 = dataframe2.values
data_2 = dataset2[20000:40000,0]
paths_old = data_2
#img_old = cv2.imread(path_old, 0)
data2_2 = dataset2[20030:40030,0]

dataframe = pandas.read_csv('roadseg_dataset_newest.csv',delimiter=",")
dataset = dataframe.values
data = dataset[20000:40000,0:6]
data2 = dataset[20030:40030,0:6]
path = data[0][0]


writer = VideoWriter('video.avi', frameSize=(1280,720))
writer.open()
for i in range(0,2000):
	path = data[i][0]
	img = cv2.imread(path)
	img = cv2.resize(img, (28,28))
	X = numpy.array(img)
	X = X.reshape(1, 3, 28, 28).astype('float32')
	X = X / 255

	path2 = data2[i][0]
	img2 = cv2.imread(path2)
	img2 = cv2.resize(img2, (28,28))
	X2 = numpy.array(img2)
	X2 = X2.reshape(1, 3, 28, 28).astype('float32')
	X2 = X2 / 255
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	score1 = loaded_model.predict(X)
	score2 = loaded_model.predict(X2)
	img_old = cv2.imread(paths_old[i])
	P_pitch = score1[0][0]
	C_pitch = data[i][2]
	#print(C_pitch)
	C_vel = data[i][4]
	t = 1
	slope1 = slope(P_pitch,C_pitch,C_vel,t)
	# slope1 = numpy.around(numpy.rad2deg(numpy.arcsin(-(P_pitch-C_pitch)/(C_vel*t))), decimals=2)
	print("Predicted slope based on IMU pitch", slope1)
	cv2.putText(img_old,"Predicted slope based on IMU pitch: %f" %slope1, bottomLeftCornerOfText, font, 	fontScale, fontColor, lineType)
	print(i)
	cv2.imwrite('saved/{}.jpg'.format(i),img_old)
	writer.write(img_old)
#cv2.destroyAllWindows()
writer.release()
