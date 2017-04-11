from operator import attrgetter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pykitti

# Change this to the directory where you store KITTI data
basedir = "/mnt/sakuradata2/datasets/kitti/tracking"

# Specify the dataset to load
sequence = "0000"

# Optionally, specify the frame range to load
frame_range = None #range(0, 20, 5)

# Load the data
dataset = pykitti.tracking(basedir, sequence, train=True, frame_range=frame_range)

def get_t(matrix):
	return matrix[:,3]

def get_xyz(array):
	return zip(*[(x[0], x[1], x[2]) for x in array])

# Load image data
#dataset.load_rgb(format='cv2')   # Loads images as uint8 with BGR ordering
dataset.load_oxts()
dataset.load_calib()
dataset.load_labels()

imu2cam = dataset.calib.Tr_imu_velo.dot(dataset.calib.Tr_velo_cam)
cam2imu = np.linalg.inv(imu2cam)

camera_height = 1.65

# count number of true tracks
num_true_tracks = max(dataset.labels, key=attrgetter("track_id")).track_id

car = []

for oxt in dataset.oxts:
	Tr_w_imu = oxt.T_w_imu
	car.append(get_t(Tr_w_imu))

tracks = []

for trk in range(num_true_tracks):

	track = []

	for i in range(len(dataset.labels)):
		lbl = dataset.labels[i]
		
		if int(lbl.track_id) != trk:
			continue

		Tr_w_imu = dataset.oxts[lbl.frame].T_w_imu

		obj = np.array([lbl.loc_x, lbl.loc_y-camera_height, lbl.loc_z, 1.0])
		obj = cam2imu.dot(obj.T)

		obj = Tr_w_imu.dot(obj)
		obj /= obj[3]
		track.append(obj)

	tracks.append(track)

fig = plt.figure()

ax = fig.add_subplot(211, projection="3d")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("height [m]")

x,y,z = get_xyz(car)
ax.plot(x,y,z, ls="dashed")

for t in tracks:
	tx,ty,tz = get_xyz(t)
	ax.plot(tx,ty,tz, color="red")

ax2 = fig.add_subplot(212)
ax2.grid()

ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.plot(x,y, ls="dashed")

for t in tracks:
	tx,ty,tz = get_xyz(t)
	ax2.plot(tx,ty, color="red")

plt.tight_layout()
plt.show()