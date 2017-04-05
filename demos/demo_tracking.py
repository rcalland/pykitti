"""Example of pykitti.raw usage with OpenCV."""
import cv2
import matplotlib.pyplot as plt

import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = "/mnt/sakuradata2/datasets/kitti/tracking"

# Specify the dataset to load
sequence = "0000"

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
dataset = pykitti.tracking(basedir, sequence, frame_range)

# Load image data
dataset.load_rgb(format='cv2')   # Loads images as uint8 with BGR ordering
dataset.load_oxts()

# Do some stereo processing
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disp_rgb = stereo.compute(
    cv2.cvtColor(dataset.rgb[0].left, cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(dataset.rgb[0].right, cv2.COLOR_BGR2GRAY))

# Display some data
f, ax = plt.subplots(2, 1) #, figsize=(15, 5))

ax[0].imshow(cv2.cvtColor(dataset.rgb[0].left, cv2.COLOR_BGR2RGB))
ax[0].set_title('Left RGB Image (cam2)')

disp = ax[1].imshow(disp_rgb, cmap='viridis')
cbar = f.colorbar(disp, orientation='horizontal')
ax[1].set_title('RGB Stereo Disparity')

plt.show()
