# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

import numpy as np
import cv2
import os
import queue
from utils import CameraPoses
from threaded_reader import ThreadedVideoCapture
import pyqtgraph as pg
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

app = pg.mkQApp()
phone_ips = {'work': "10.0.49.50",
             'home': "192.168.1.21"}

frame1 = cv2.imread(r"C:\Users\magla\Pictures\PXL_20221023_192919429.jpg")
frame2 = cv2.imread(r"C:\Users\magla\Pictures\PXL_20221023_192924868.jpg")
source = 'stream'

# Camera matrix made with video resolution of 1280x720
camera_matrix = np.array([[1.01037E3, 0.0,       6.35476E2],
                          [0.0,       1.01035E3, 3.54001E2],
                          [0.0,       0.0,       1.0]])
phone_ip = phone_ips['home']
capture_port = 8080
sensor_port = 5000

camtrack = CameraPoses(camera_matrix)

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
if source == 'stream':
    capture = ThreadedVideoCapture(rf"http://{phone_ip}:{capture_port}/video", cv2.CAP_FFMPEG)
    if not capture._capture.isOpened():
        print(f'Cannot open RTSP stream.')
        exit(-1)

# Begin viewing and processing the video stream
process_times = np.empty(shape=(30,), dtype=float)
ransac_percent = np.empty(shape=(30,), dtype=float)

counter = 0
ret = True
last_stats_update = 0
while True:
    if source == 'stream':
        try:
            ret, frame = capture.read()
        except queue.Empty:
            ret, frame = False, None
    else:
        frame = frame1 if counter % 2 == 0 else frame2
    if ret:
        cv2.imshow("Frame", frame)
        camtrack.track(frame)
        if cv2.waitKey(1) == ord("s"):
            cv2.imwrite(rf"C:\Users\magla\Pictures\phone3dscanner\{counter}.jpg", frame)
    if ret is None:
        break
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("f"):
        capture.poll_rate = (capture.poll_rate + 10) % 50 + 5
    counter += 1
    if time.perf_counter() - last_stats_update > 1:
        if capture.actual_poll_rate is not None and capture.fps is not None:
            logging.info(f"Real poll rate: {capture.actual_poll_rate:.1f}, FPS: {capture.fps:.1f}")
            last_stats_update = time.perf_counter()

if source == 'stream':
    capture.release()

cv2.destroyAllWindows()

# with PhoneSensorsClient(phone_ip, sensor_port) as client:
#     for packet in client:
#         print(packet)
