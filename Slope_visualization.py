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
from tqdm import *
from skvideo.io import VideoWriter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def slope(pitch_1,pitch_2,velocity,time):
    slope = numpy.around(numpy.rad2deg(numpy.arcsin(-(pitch_1-pitch_2)/(velocity*time))), decimals=2)
    return slope
ts = []

for i in np.arange(0,1000,0.033):
	if len(ts)<20000:
		ts.append(i)
#plt.plot(ts)
#plt.show()

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
data = dataset[20000:40000,0:7]

imu_p_smooth = dataset[20030:40030,2]
# imu_p_smooth = dataset[20000:40000,2]
plt.plot(ts,imu_p_smooth,label='Unmodified pitch')

#SG filter
# size = 1001
# imu_p_smooth = savgol_filter(imu_p_smooth,size,3)
# plt.plot(ts,imu_p_smooth,label='SG filter (1001 values, 3rd degree)')

#MA filter
N = 500
imu_p_smooth_n = np.zeros(int(N/2)-1)
imu_p_smooth_nn = np.zeros(int(N/2))
imu_p_smooth_t = np.convolve(imu_p_smooth, np.ones((N,))/N, mode='valid')
# np.append(imu_p_smooth,imu_p_smooth_t)
imu_p_smooth = np.concatenate((imu_p_smooth_n,imu_p_smooth_t),axis=0)
imu_p_smooth = np.concatenate((imu_p_smooth,imu_p_smooth_nn),axis=0)
# plt.plot(ts[:20000-N+1],imu_p_smooth,label='Moving-avg filter (30 values)')
plt.plot(ts,imu_p_smooth,label='Moving-avg filter (30 values)')


plt.xlabel('Time(Seconds)')
plt.ylabel('Slope')
plt.legend()
plt.show()
data2 = dataset[20030:40030,0:6]
path = data[0][0]


writer = VideoWriter('video.avi', frameSize=(1280,720))
writer.open()

for i in tqdm(range(0,2000)):
	#C_pitch = data[i][2]
	C_pitch = imu_p_smooth[i]
	#print(C_pitch)
	C_vel = data[i][4]
	t = 1
	#slope1 = slope(P_pitch,C_pitch,C_vel,t)
	# slope1 = numpy.around(numpy.rad2deg(numpy.arcsin(-(P_pitch-C_pitch)/(C_vel*t))), decimals=2)
	#print("Predicted slope based on IMU pitch", slope1)
	img_old = cv2.imread(paths_old[i])	
	cv2.putText(img_old,"Predicted IMU pitch for next second: %f" %C_pitch, bottomLeftCornerOfText, font, 	fontScale, fontColor, lineType)
	#cv2.imwrite('saved/{}.jpg'.format(i),img_old)
	writer.write(img_old)
#cv2.destroyAllWindows()
writer.release()
