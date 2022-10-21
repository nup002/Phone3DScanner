# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

import numpy as np
import cv2
import os
from time import perf_counter
from utils import CameraPoses
import pyqtgraph as pg
app = pg.mkQApp()

camera_matrix = np.array([[3.257E3, 0, 1.552E3],
                          [0, 3.271E3, 2.020E3],
                          [0, 0, 1]])

phone_ip = "10.0.49.50"
capture_port = 8080
sensor_port = 5000

camtrack = CameraPoses(camera_matrix)

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
capture = cv2.VideoCapture(rf"http://{phone_ip}:{capture_port}/video", cv2.CAP_FFMPEG)
if not capture.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

# Begin viewing and processing the video stream
process_times = np.empty(shape=(30,), dtype=float)
ransac_percent = np.empty(shape=(30,), dtype=float)

while True:
    ret, frame = capture.read()
    if ret:
        camtrack.track(frame)
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

# with PhoneSensorsClient(phone_ip, sensor_port) as client:
#     for packet in client:
#         print(packet)
