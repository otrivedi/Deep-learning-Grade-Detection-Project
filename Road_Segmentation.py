import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import os
import glob
import csv
import pandas

def invert_img(img):
    img = (255-img)
    return img
def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

if not os.path.exists("dataset"):
    os.system("mkdir dataset")

fpath = sorted(glob.glob("data/*"), key=sortKeyFunc)

count = 0
for f in fpath:
    img = cv2.imread(f)
    img = imutils.resize(img, height = 300)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    thresh = invert_img(thresh)

    # noise removal
    kernel = np.ones((10,10), np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 4)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    #sure_bg = cv2.morphologyEx(sure_bg, cv2.MORPH_TOPHAT, kernel)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    '''
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
    img = cv2.Canny(imgray,200,500)
    '''
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    road = cv2.subtract(thresh,sure_bg)

    # r = 100.0 / road.shape[1]
    # dim = (100, int(road.shape[0] * r))

    road = cv2.resize(road,(720,720))
    count += 1
    print(count)
    cv2.imwrite("dataset/%d.jpg" %count , road)

# image_path = glob.glob("dataset/*/*")
# dataset = pandas.read_csv('new_data_fps30.csv', header=1)
# ts=dataset.iloc[:, 7]
# # ilabel = [os.path.basename(os.path.dirname(x)) for x in image_path]
# rows = zip(image_path,ts)
# with open("dataset/roadseg_dataset.csv", "w") as f:
#     writer = csv.writer(f)
#     for row in rows:
#         writer.writerow(row)
