import time

import depthai as dai
import math
import numpy as np
import cv2
import spatial

# Create ROI rect
ROI = dai.Rect(dai.Point2f(0.5, 0.5), dai.Point2f(0.55, 0.55))

pipeline = dai.Pipeline()

# Setup core nodes
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
slc = pipeline.create(dai.node.SpatialLocationCalculator)

# Initialize properties for mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Initialize properties for stereo node
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Initialize properties for spatial location calculator
slcConfig = dai.SpatialLocationCalculatorConfigData()
slcConfig.depthThresholds.lowerThreshold = 100
slcConfig.depthThresholds.upperThreshold = 15000
slcConfig.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
slcConfig.roi = ROI
slc.initialConfig.addROI(slcConfig)

# Link nodes
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(slc.inputDepth)

# Create outputs
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDisparity.setStreamName("disparity")
stereo.disparity.link(xoutDisparity.input)

xoutImage = pipeline.create(dai.node.XLinkOut)
xoutImage.setStreamName("image")
stereo.rectifiedRight.link(xoutImage.input)

xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutSpatialData.setStreamName("spatialData")
slc.out.link(xoutSpatialData.input)


with dai.Device(pipeline, dai.DeviceInfo("194430100112801300")) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qDisparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    qImage = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    qSpatialData = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

    # Setup on-host spatial calculator
    scalc = spatial.SpatialCalculator(device, monoRight.getResolutionSize(), stereo.initialConfig.getMaxDisparity())

    # Setup named windows
    x = 0
    for n in ["disparity", "image"]:
        cv2.namedWindow(n, cv2.WINDOW_NORMAL)
        cv2.moveWindow(n, x, 0)
        x += 640

    while True:
        inDisparity = qDisparity.get()
        inImage = qImage.get()
        inSpatialData = qSpatialData.get()

        # Extract disparity frame
        disparityFrame = inDisparity.getCvFrame()
        imageFrame = inImage.getCvFrame()

        # Update depth map for spatial calculator
        scalc.update_depth(disparityFrame)

        # Calculate rect on the disparity frame
        rect = (int(ROI.topLeft().x * disparityFrame.shape[1]),
                int(ROI.topLeft().y * disparityFrame.shape[0]),
                int(ROI.bottomRight().x * disparityFrame.shape[1]),
                int(ROI.bottomRight().y * disparityFrame.shape[0]))

        # Calculate spatial coordinate for ROI using on-host spatial calculator
        hx, hy, hz = scalc.calculate(rect[0], rect[1], rect[2], rect[3])
        hdist = math.sqrt(hx**2 + hy**2 + hz**2) / 1000

        # Pull the spatial data from the spatial location calculator
        scoords = inSpatialData.spatialLocations[0].spatialCoordinates
        sx, sy, sz = scoords.x, scoords.y, scoords.z
        sdist = math.sqrt(sx**2 + sy**2 + sz**2) / 1000

        # Render the spatial data on the image frame
        cv2.putText(imageFrame, f"On-Host: {hdist:.2f} m", (10, imageFrame.shape[0]-80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.putText(imageFrame, f"On-Device: {sdist:.2f} m", (10, imageFrame.shape[0]-60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        # Render the ROI area
        cv2.rectangle(imageFrame,
                      (int(ROI.topLeft().x * imageFrame.shape[1]),
                       int(ROI.topLeft().y * imageFrame.shape[0])),
                      (int(ROI.bottomRight().x * imageFrame.shape[1]),
                       int(ROI.bottomRight().y * imageFrame.shape[0])), (255, 255, 0), 4)

        disparityFrame = cv2.applyColorMap((disparityFrame / disparityFrame.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.rectangle(disparityFrame,
                        (int(ROI.topLeft().x * disparityFrame.shape[1]),
                         int(ROI.topLeft().y * disparityFrame.shape[0])),
                        (int(ROI.bottomRight().x * disparityFrame.shape[1]),
                         int(ROI.bottomRight().y * disparityFrame.shape[0])), (255, 255, 0), 4)

        cv2.imshow("image", imageFrame)
        cv2.imshow("disparity", disparityFrame)

        if cv2.waitKey(1) == ord("q"):
            break
