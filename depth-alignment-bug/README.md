
# Getting started

1. Setup a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install opencv-python depthai==2.21.1`
3. Run the example: `python main.py`

# The Issue

The Yolo spatial location calculator is returning incorrect results when the mono cameras are configured to use
THE_720 resolution; the correct result is returned when THE_400 resolution is used. In the included example PNGs,
the correct distance of the TV and chair is ~2.8m and ~1.9m respectively. The distance is roughly halved when using
the higher resolution setting for the mono cameras.

Note that if we DISABLE depth alignment to the RGB camera, the correct distance is returned, albiet with
many warnings about inference not using correct alignment.