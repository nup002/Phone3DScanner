# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

import numpy as np
import cv2
import os
import queue
import time
from utils import CameraPoses
from threaded_reader import start_threaded_reader
import pyqtgraph as pg

app = pg.mkQApp()
phone_ips = {'work': "10.0.49.50",
             'home': "192.168.1.21"}

frame1 = cv2.imread(r"C:\Users\magla\Pictures\PXL_20221023_192919429.jpg")
frame2 = cv2.imread(r"C:\Users\magla\Pictures\PXL_20221023_192924868.jpg")
source = 'stream'
camera_matrix = np.array([[2.248E3, 0.0,     1.347E3],
                          [0.0,     2.23799E3, 9.93E2],
                          [0.0,     0.0,     1.0]])
phone_ip = phone_ips['home']
capture_port = 8080
sensor_port = 5000

camtrack = CameraPoses(camera_matrix)

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
if source == 'stream':
    capture = cv2.VideoCapture(rf"http://{phone_ip}:{capture_port}/video", cv2.CAP_FFMPEG)
    if not capture.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)
    frames_queue = start_threaded_reader(capture)

# Begin viewing and processing the video stream
process_times = np.empty(shape=(30,), dtype=float)
ransac_percent = np.empty(shape=(30,), dtype=float)

skip_frames = 1
counter = 0
ret = True
while True:
    if source == 'stream':
        try:
            ret, frame = frames_queue.get(block=False)
        except queue.Empty:
            ret, frame = False, None
    else:
        frame = frame1 if counter % 2 == 0 else frame2
    if ret and counter % skip_frames == 0:
        cv2.imshow("Frame", frame)
        time.sleep(0.5)
        #camtrack.track(frame)
    if cv2.waitKey(1) == ord("q"):
        break
    counter += 1

capture.release()
cv2.destroyAllWindows()

# with PhoneSensorsClient(phone_ip, sensor_port) as client:
#     for packet in client:
#         print(packet)
