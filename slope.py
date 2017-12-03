import numpy as np
import pandas
import matplotlib.pyplot as plt

dataset = pandas.read_csv('roadseg_dataset_newest.csv', header=1)

gps_al = dataset.iloc[:, 3]
gps_al_train = dataset.iloc[10000:36800, 3]
gps_al_val = dataset.iloc[36800:50000, 3]
gps_al_test = dataset.iloc[55000:60000, 3]

pitch = dataset.iloc[:, 2]
pitch_train = dataset.iloc[10000:36800, 2]
pitch_val = dataset.iloc[36800:50000, 2]
pitch_test = dataset.iloc[55000:60000, 2]

time_train = dataset.iloc[10000:36800,1]
time_val = dataset.iloc[36800:50000, 1]
time_test = dataset.iloc[55000:60000, 1]
time = dataset.iloc[:,1]
time = time/1000
time_train = time_train/1000
time_val = time_val/1000
time_test = time_test/1000

plt.plot(time, gps_al)
plt.plot(time_train, gps_al_train, label = 'Training')
plt.plot(time_val, gps_al_val, label = 'Validation')
plt.plot(time_test, gps_al_test, label = 'Testing')
# plt.legend([line1, line2, line3], ['Training', 'Validation', 'Testing'])
plt.xlabel("Time in sec")
plt.ylabel("GPS Altitude")
plt.title("Dataset segmentation")
plt.show()

plt.plot(time, pitch)
plt.plot(time_train, pitch_train, label = 'Training')
plt.plot(time_val, pitch_val, label = 'Validation')
plt.plot(time_test, pitch_test, label = 'Testing')
# plt.legend([line1, line2, line3], ['Training', 'Validation', 'Testing'])
plt.xlabel("Time in sec")
plt.ylabel("IMU Pitch")
plt.title("Dataset segmentation")
plt.show()
