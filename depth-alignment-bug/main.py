import time

import depthai as dai
import math
import numpy as np
import cv2
import blobconverter

TARGET_FPS=25

pipeline = dai.Pipeline()

# Setup core nodes
rgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sdn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

# Initialize properties for color camera
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
rgb.setIspScale(4, 29)
rgb.setPreviewSize(416, 416)
rgb.setInterleaved(False)
rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
rgb.setFps(TARGET_FPS)

# Initialize properties for mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(TARGET_FPS)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setFps(TARGET_FPS)

# Initialize properties for stereo node
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Initialize detection network
sdn.setBlob(blobconverter.from_zoo("yolov6n_coco_416x416", zoo_type="depthai"))
sdn.setConfidenceThreshold(0.8)
sdn.setBoundingBoxScaleFactor(0.8)
sdn.setDepthLowerThreshold(500)
sdn.setDepthUpperThreshold(15000)
sdn.setCoordinateSize(4)
sdn.setNumClasses(80)
sdn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
sdn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
sdn.setIouThreshold(0.6)
sdn.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.MEDIAN)
sdn.setNumInferenceThreads(2)
sdn.input.setBlocking(False)

# Link mono nodes into stereo depth
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Link rgb + stereo into detection system
rgb.preview.link(sdn.input)
stereo.depth.link(sdn.inputDepth)

# Link all output queues
xoutImage = pipeline.create(dai.node.XLinkOut)
xoutImage.setStreamName("image")
rgb.isp.link(xoutImage.input)

xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutSpatialData.setStreamName("spatialData")
sdn.out.link(xoutSpatialData.input)


with dai.Device(pipeline, dai.DeviceInfo("194430100112801300")) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qImage = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    qSpatialData = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

    while True:
        inImage = qImage.get()
        inSpatialData = qSpatialData.get()

        imageFrame = inImage.getCvFrame()

        # Walk over detections and draw on image frame
        for detection in inSpatialData.detections:
            # Get the spatial location of the detection
            spatialLocation = detection.spatialCoordinates

            # Get the distance of the detection
            dist = math.sqrt(spatialLocation.x**2 + spatialLocation.y**2 + spatialLocation.z**2) / 1000 # in meters

            # Draw the bounding box
            cv2.rectangle(imageFrame,
                          (int(detection.xmin * imageFrame.shape[1]),
                           int(detection.ymin * imageFrame.shape[0])),
                          (int(detection.xmax * imageFrame.shape[1]),
                           int(detection.ymax * imageFrame.shape[0])), (255, 255, 0), 4)

            # Draw the label
            cv2.putText(imageFrame, f"{dist:.2f} - {monoLeft.getResolutionHeight()}x{monoLeft.getResolutionWidth()}",
                        (int(detection.xmin * imageFrame.shape[1]),
                         int(detection.ymin * imageFrame.shape[0]) + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))


        cv2.imshow("image", imageFrame)

        if cv2.waitKey(1) == ord("q"):
            break
