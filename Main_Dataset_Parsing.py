import numpy as np
import cv2
import pandas

cap = cv2.VideoCapture('video_front.mp4')
dataset = pandas.read_csv('data_fps30.csv', header=1)
ts=dataset.iloc[:, 0]
count = 0
for i in ts:
    cap.set(cv2.CAP_PROP_POS_MSEC,i)
    success,image =cap.read()
    count += 1
    success = True
    if success:
    #   success,image = cap.read()
      print('Read a new frame: ', success)
      cv2.imwrite("data/frame%d.jpeg" % count, image)     # save frame as JPEG file
      cv2.waitKey()
print('The total number of images are', count)
