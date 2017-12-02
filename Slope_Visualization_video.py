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
from PIL import Image


def slope(pitch_1,pitch_2,velocity,time):
    slope = numpy.around(numpy.rad2deg(numpy.arcsin(-(pitch_1-pitch_2)/(velocity*time))), decimals=2)
    return slope
ts = []

for i in np.arange(0,6000,0.033):
	if len(ts)<5000:
		ts.append(i)
#plt.plot(ts)
#plt.show()

font                   = cv2.FONT_HERSHEY_TRIPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1


K = []
S=55000
E=60000
SS=55060
EE=60060
dataframe2 = pandas.read_csv('road_dataset_newest.csv')
dataset2 = dataframe2.values
data_2 = dataset2[S:E,0]
paths_old = data_2
#img_old = cv2.imread(path_old, 0)
data2_2 = dataset2[SS:EE,0]

dataframe = pandas.read_csv('roadseg_dataset_newest.csv',delimiter=",")
dataset = dataframe.values
data = dataset[S:E,0:7]

imu_p_smooth = dataset[S:E,2]
plt.plot(ts,imu_p_smooth,label='Unmodified pitch')


# imu_p_smooth = dataset[20000:40000,2]


#SG filter
size = 91
imu_p_smooth = savgol_filter(imu_p_smooth,size,2)
plt.plot(ts,imu_p_smooth,label='SG filter ({} values, 3rd degree)'.format(size))

#MA filter
N = 90
imu_p_smooth_ma = dataset[S-N:E+N,2]
#imu_p_smooth_nn = np.zeros(int(N/2))
imu_p_smooth_t = np.convolve(imu_p_smooth_ma, np.ones((N,))/N, mode='valid')
# np.append(imu_p_smooth,imu_p_smooth_t)
# imu_p_smooth = np.concatenate((imu_p_smooth_ma,imu_p_smooth_t),axis=0)
# imu_p_smooth = np.concatenate((imu_p_smooth,imu_p_smooth_nn),axis=0)
# plt.plot(ts[:20000-N+1],imu_p_smooth,label='Moving-avg filter (30 values)')
plt.plot(ts,imu_p_smooth_t[int(N/2):int(len(imu_p_smooth_t)-(N/2))-1],label='Moving-avg filter ({} values)'.format(N))


plt.xlabel('Time(Seconds)')
plt.ylabel('Slope')
plt.legend()
plt.show()
data2 = dataset[SS:EE,0:6]
path = data[0][0]

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    #overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    #background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

writer = VideoWriter('video.avi', frameSize=(1280,720))
writer.open()

for i in tqdm(range(0,1000)):
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
	json_file = open("model.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	score1 = loaded_model.predict(X)
	score2 = loaded_model.predict(X2)
	P_pitch = score1[0][0]
	C_pitch = data[i][2]
	#print(C_pitch)
	C_vel = data[i][4]
	t = 1
	#slope1 = slope(P_pitch,C_pitch,C_vel,t)
	# slope1 = numpy.around(numpy.rad2deg(numpy.arcsin(-(P_pitch-C_pitch)/(C_vel*t))), decimals=2)
	#print("Predicted slope based on IMU pitch", slope1)
	img = Image.open(paths_old[i])
	img2 = Image.open('car.jpg')
	img.paste(img2,(15,15))
	img.save('output.jpg')
	img_old = cv2.imread('output.jpg')	
	cv2.putText(img_old,"Predicted road slope (IMU pitch): %f" %C_pitch, bottomLeftCornerOfText, font, 	fontScale, fontColor, lineType)
	height = int(90-(C_pitch*5))
	if height<90:
		status = "Accelerate"
	else:
		status = "Decelerate"
	cv2.rectangle(img_old,(10,10),(400,175),(0,0,255),3)
	cv2.line(img_old,(205,90),(370,height),(255,0,0),3)
	cv2.putText(img_old,"Vehicle mode: {}".format(status),(20,160),font,fontScale,(255,0,0), 1)
	#cv2.imwrite('saved/{}.jpg'.format(i),img_old)
	#cv2.imwrite('{}.jpg'.format(i),img_old)
	#im.save(
	writer.write(img_old)
#cv2.destroyAllWindows()
writer.release()
