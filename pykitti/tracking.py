"""Provides 'tracking', which loads and parses the KITTI tracking dataset."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

import pykitti.utils as utils

__author__ = "Richard Calland"
__email__ = "calland@preferred.jp"

class tracking:

    def __init__(self, base_path, sequence, train=True, frame_range=None):

        self.sequence = sequence
        self.train = train
        if self.train:
            self.base_path = os.path.join(base_path, "training")
        else:
            self.base_path = os.path.join(base_path, "testing")

        self.oxts_path = os.path.join(self.base_path, "oxts")
        self.calib_path = os.path.join(self.base_path, "calib")

        if self.train:
            self.label_path = os.path.join(self.base_path, "label_02")
        else:
            self.label_path = None

        self.frame_range = frame_range

    def load_oxts(self):
        """Load OXTS data from file."""
        print('Loading OXTS data from ' + self.sequence + '...')

        # Extract the data from each OXTS packet
        OxtsPacket = namedtuple('OxtsPacket',
                                'lat, lon, alt, ' +
                                'roll, pitch, yaw, ' +
                                'vn, ve, vf, vl, vu, ' +
                                'ax, ay, az, af, al, au, ' +
                                'wx, wy, wz, wf, wl, wu, ' +
                                'pos_accuracy, vel_accuracy, ' +
                                'navstat, numsats, ' +
                                'posmode, velmode, orimode')

        filename = os.path.join(self.oxts_path, "{}.txt".format(self.sequence))

        oxts_packets = []
        #for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                data = OxtsPacket(*line)
                oxts_packets.append(data)

        # restrict to the frame selection
        if self.frame_range:
            oxts_packets = [oxts_packets[i] for i in self.frame_range]

        # Precompute the IMU poses in the world frame
        T_w_imu = utils._poses_from_oxts(oxts_packets)

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
        self.oxts = []
        for (p, T) in zip(oxts_packets, T_w_imu):
            self.oxts.append(OxtsData(p, T))

        print('done.')

    def load_rgb(self, **kwargs):
        """Load RGB stereo images from file.

        Setting imformat='cv2' will convert the images to uint8 and BGR for
        easy use with OpenCV.
        """

        print('Loading color images from ' + self.sequence + '...')

        imL_path = os.path.join(self.base_path, 'image_02', self.sequence, '*.png')
        imR_path = os.path.join(self.base_path, 'image_03', self.sequence, '*.png')

        imL_files = sorted(glob.glob(imL_path))
        imR_files = sorted(glob.glob(imR_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            imL_files = [imL_files[i] for i in self.frame_range]
            imR_files = [imR_files[i] for i in self.frame_range]

        print('Found ' + str(len(imL_files)) + ' image pairs...')

        self.rgb = utils.load_stereo_pairs(imL_files, imR_files, **kwargs)

        print('done.')
