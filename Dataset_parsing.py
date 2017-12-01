import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1

if __name__=="__main__":
    print("aba")
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)

# # create a folder to store extracted images
# import os
# import cv2
#
# folder = 'video_front'
# # os.mkdir(folder)
# cap = cv2.VideoCapture("video_front.avi")
# success,image = cap.read()
# print(success)
# count = 0
# while success:
#     print('ok')
#     cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
#     count += 1
# print("{} images are extacted in {}.".format(count,folder))

# import numpy as np
# import cv2
# import os
#
# folder = 'video_front'
# os.mkdir(folder)
# cap = cv2.VideoCapture('video_front.mp4')
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc,100.0, (640,480))
#
# count = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)
#
#         # write the flipped frame
#         out.write(frame)
#
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         count += 1
#         cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
#
#     else:
#         break
#
# print("{} images are extacted in {}.".format(count,folder))
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()
