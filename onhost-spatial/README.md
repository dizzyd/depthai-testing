
# Getting started

1. Setup a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install opencv-python depthai==2.21.0`
3. Run the example: `python main.py`

# The Issue

As of depthai 2.21.0, the spatial location calculator is returning incorrect results. In the example.png image,
you can see the bounding box of the area we're using to calculate the spatial location. Using a laser measurement tool,
the cone is 1.88m away from the camera. We can see that using the disparity map directly and calculating on-host, we get
a number within 10cm of the laser measurement. However, using the spatial location calculator, we get a number that is
more off by at least a full meter.