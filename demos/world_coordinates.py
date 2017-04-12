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

# Use the training data? If false, will use test data
train = True

# Load the data
dataset = pykitti.tracking(basedir, sequence, train=train)

# get the translation vector
def get_t(matrix):
	return matrix[:,3]

# convert a list of numpy arrays into 3 lists containing x,y,z coords
def get_xyz(array):
	return zip(*[(x[0], x[1], x[2]) for x in array])

# Load image data
#dataset.load_rgb(format='cv2')   # Loads images as uint8 with BGR ordering
dataset.load_oxts()
dataset.load_calib()
dataset.load_labels()

# count number of true tracks
num_true_tracks = max(dataset.labels, key=attrgetter("track_id")).track_id

# draw the car path
car = []

for oxt in dataset.oxts:
	Tr_w_imu = oxt.T_w_imu
	car.append(get_t(Tr_w_imu))

# draw track paths
tracks = []

for trk in range(num_true_tracks):

	track = []

	for i in range(len(dataset.labels)):
		lbl = dataset.labels[i]
		
		if int(lbl.track_id) != trk:
			continue

		T_w_imu = dataset.oxts[lbl.frame].T_w_imu

		# get object location in image coordinates, convert to camera coordinates
		pos_x = 0.5*(lbl.bbox_right - lbl.bbox_left) + lbl.bbox_left
		pos_y = lbl.bbox_bottom
		pos_z = lbl.loc_z

		cam_pos = np.array([pos_x*pos_z, pos_y*pos_z, pos_z, 1.0])
		cam_pos = np.matmul(dataset.calib.T_imgL_cam.T, cam_pos.T)
		obj = dataset.calib.T_cam_imu.dot(cam_pos)

		# transform from cam to world coords
		obj = T_w_imu.dot(obj)
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
