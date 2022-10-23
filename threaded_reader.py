#!/usr/bin/env python3

"""
Author: 
"""

import cv2
import threading
import queue

def reader(capture, q: queue.LifoQueue):
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            if q.full():
                q.get(block=False)
            q.put((ret, frame), block=False)

def start_threaded_reader(capture: cv2.VideoCapture):
    q = queue.LifoQueue(maxsize=1)
    threaded_reader = threading.Thread(target=reader, args=(capture, q), daemon=True)
    threaded_reader.start()
    return q