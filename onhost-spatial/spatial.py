
import depthai as dai
import numpy as np
import math

class SpatialCalculator:
    def __init__(self, device, resolution, max_disparity):
        calib = device.readCalibration()
        self.baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution)
        self.focalLength = intrinsics[0][0]
        self.scaleFactor = self.baseline * self.focalLength * (max_disparity / 95)
        self.hfov = np.deg2rad(calib.getFov(dai.CameraBoardSocket.LEFT))
        print(f"Focal length: {self.focalLength}, baseline: {self.baseline}, scaleFactor: {self.scaleFactor} hfov: {self.hfov}")

    def calc_angle(self, coord):
        return math.atan(math.tan(self.hfov / 2.0) * coord / (self.depth.shape[1] / 2.0))

    def update_depth(self, disparity):
        # Calculate the depth frame
        self.depth = (self.scaleFactor / disparity).astype(np.uint16)

    def calculate(self, xmin, ymin, xmax, ymax):
        # Extract the region of interest from the detection
        roi = self.depth[ymin:ymax, xmin:xmax]

        # Get the centroid of the ROI
        x = int((xmin + xmax) / 2)
        y = int((ymin + ymax) / 2)

        midW = int(self.depth.shape[1] / 2) # middle of the depth img width
        midH = int(self.depth.shape[0] / 2) # middle of the depth img height
        bb_x_pos = x - midW
        bb_y_pos = y - midH

        # print(f"bb_x_pos: {bb_x_pos}, bb_y_pos: {bb_y_pos}")

        angle_x = math.atan(math.tan(self.hfov / 2.0) * bb_x_pos / (self.depth.shape[1] / 2.0))
        angle_y = math.atan(math.tan(self.hfov / 2.0) * bb_y_pos / (self.depth.shape[0] / 2.0))

        depth = np.median(roi)

        # Calculate the average depth of the ROI
        return depth * math.tan(angle_x), -depth * math.tan(angle_y), depth