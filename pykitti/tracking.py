"""Provides 'tracking', which loads and parses the KITTI tracking dataset."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

import pykitti.utils as utils

class tracking:

    def __init__(self, base_path, sequence, train=True, frame_range=None):

        self.sequence = sequence
        self.train = train

        if self.train:
            print("Using training dataset.")
            self.base_path = os.path.join(base_path, "training")
            self.label_path = os.path.join(self.base_path, "label_02")
        else:
            print("Using testing dataset.")
            self.base_path = os.path.join(base_path, "testing")
            self.label_path = None

        self.oxts_path = os.path.join(self.base_path, "oxts")
        self.calib_path = os.path.join(self.base_path, "calib")

        self.frame_range = frame_range

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.calib_path, "{}.txt".format(self.sequence))

        filedata = utils.read_calib_file(calib_filepath)

        P2 = utils.pad3x4_to_4x4(np.reshape(filedata["P2"], (3,4)))

        R_rect = np.reshape(filedata["R_rect"], (3,3))
        R_rect = utils.pad3x4_to_4x4(np.hstack((R_rect, np.array([0.0, 0.0, 0.0])[:, None])))

        data["P2"] = P2
        data["R_rect"] = R_rect

        # transform from left camera coordinates into left image coords
        data["T_cam_imgL"] = np.dot(P2, R_rect)
        data["T_imgL_cam"] = np.linalg.inv(data["T_cam_imgL"])

        T_velo_cam = np.reshape(filedata["Tr_velo_cam"], (3,4))
        T_imu_velo = np.reshape(filedata["Tr_imu_velo"], (3,4))

        data["T_velo_cam"] = utils.pad3x4_to_4x4(T_velo_cam)
        data["T_imu_velo"] = utils.pad3x4_to_4x4(T_imu_velo)
        # convert GPS/IMU coords into camera via velo
        data["T_imu_cam"] = data["T_velo_cam"].dot(data["T_imu_velo"])
        data["T_cam_imu"] = np.linalg.inv(data["T_imu_cam"])

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def load_labels(self):
        print(self.label_path)

        if self.label_path is None:
            print("No labels found.")
            return

        print("Loading labels from " + self.sequence + "...")

        FrameLabel = namedtuple("FrameLabel",
                                "frame, track_id, type, truncated, " +
                                "occluded, alpha, " +
                                "bbox_left, bbox_top, bbox_right, bbox_bottom, " +
                                "height, width, length, " +
                                "loc_x, loc_y, loc_z, " +
                                "rotation_y") #, score")

        label_seqpath = os.path.join(self.label_path, "{}.txt".format(self.sequence))

        self.labels = []

        with open(label_seqpath, "r") as f:
            for line in f.readlines():
                line = line.split()

                # convert from string
                line[:2] = [int(float(x)) for x in line[:2]]
                line[3:5] = [int(float(x)) for x in line[3:5]]
                line[5:] = [float(x) for x in line[5:]]

                data = FrameLabel(*line)
                self.labels.append(data)

        # restrict to the frame selection

        if self.frame_range:
            self.labels = [x for x in self.labels if x.frame in self.frame_range]

        print("done.")

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
        # TODO: enable this feature
        #if self.frame_range:
        #    oxts_packets = [oxts_packets[i] for i in self.frame_range]

        # Precompute the IMU poses in the world frame
        T_imu_w = utils.poses_from_oxts(oxts_packets)

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple('OxtsData', 'packet, T_imu_w')
        self.oxts = []
        for (p, T) in zip(oxts_packets, T_imu_w):
            self.oxts.append(OxtsData(p, T))

        print('done. {} frames.'.format(len(self.oxts)))

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
        self.img_shape = self.rgb[0].left.shape

        print('done.')
