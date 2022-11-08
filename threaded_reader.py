#!/usr/bin/env python3
"""
Author: Magne Lauritzen
"""
import cv2
import threading
import queue
import time
from typing import Any, Optional, Tuple
import logging
from enum import Enum, auto
from collections import deque
from statistics import mean
import numpy as np

# noinspection PyArgumentList
class InputVariables(Enum):
    TIMEOUT = auto()
    POLLRATE = auto()
    QUIT = auto()

class OutputVariables(Enum):
    POLLRATE = auto()
    FPS = auto()


def queue_put(q: queue.LifoQueue, data: Any):
    if q.full():
        q.get(block=False)
    q.put(data, block=False)


def accurate_delay(delay):
    """
    Function to provide accurate time delay
    """
    _ = time.perf_counter() + delay
    while time.perf_counter() < _:
        pass

def reader(capture: cv2.VideoCapture, video_queue: queue.LifoQueue, out_queue: queue.LifoQueue,
           in_queue: queue.Queue, timeout: float, poll_rate: float):
    poll_period_deque = deque()
    frame_period_deque = deque()
    if capture.isOpened():
        poll_period = 1 / poll_rate
        time_since_frame = 0
        last_emit_timestamp = 0
        prev_frame_timestamp = time.time()
        prev_read_timestamp = time.perf_counter()
        while time_since_frame < timeout:
            # Fetch new settings
            try:
                in_data: Tuple[InputVariables, any] = in_queue.get(block=False)
                if in_data[0] == InputVariables.TIMEOUT:
                    timeout = in_data[1]
                    logging.info(f"Timeout set to {timeout}")
                elif in_data[0] == InputVariables.POLLRATE:
                    if in_data[1] <= 0:
                        logging.warning(f"Attempted to set poll rate less or equal to 0: {in_data[1]}")
                    else:
                        poll_period = 1 / in_data[1]
                        logging.info(f"Poll rate set to {in_data[1]} Hz")
                if in_data[0] == InputVariables.QUIT:
                    logging.info(f"Received QUIT signal.")
                    break
            except queue.Empty:
                pass

            # Get time since last call and sleep if needed to reach poll_rate
            time_since_last_read = time.perf_counter() - prev_read_timestamp
            accurate_delay(max(0., poll_period - time_since_last_read))

            poll_period_deque.append(time.perf_counter() - prev_read_timestamp)
            # Read frame
            prev_read_timestamp = time.perf_counter()
            frame_available = capture.grab()
            if frame_available:
                this_frame_timestamp = time.perf_counter()
                frame_period_deque.append(this_frame_timestamp - prev_frame_timestamp)
                ret, frame = capture.retrieve()
                queue_put(video_queue, (ret, frame))
                prev_frame_timestamp = this_frame_timestamp
            time_since_frame = prev_read_timestamp - prev_frame_timestamp

            # Emit statistics
            if prev_read_timestamp - last_emit_timestamp > 1:
                mean_poll_period = mean(poll_period_deque)
                poll_period_deque.clear()
                mean_frame_period = mean(frame_period_deque)
                frame_period_deque.clear()
                queue_put(out_queue, {OutputVariables.POLLRATE: 1/mean_poll_period,
                                      OutputVariables.FPS: 1/mean_frame_period})
                last_emit_timestamp = prev_read_timestamp

    queue_put(video_queue, (None, None))


class ThreadedVideoCapture:
    def __init__(self, *args, timeout: float = 1, poll_rate: float = 100, **kwargs):
        self._capture = cv2.VideoCapture()
        self._video_queue = queue.LifoQueue(maxsize=1)
        self._output_queue = queue.LifoQueue(maxsize=1)
        self._input_queue = queue.Queue()
        self._timeout = timeout
        self._poll_rate = poll_rate
        self._return_data = {}
        self.threaded_reader: Optional[threading.Thread] = None
        self.open(*args, **kwargs)

    @property
    def actual_poll_rate(self):
        try:
            self._return_data = self._output_queue.get(block=False)
        except queue.Empty:
            pass
        finally:
            return self._return_data.get(OutputVariables.POLLRATE, None)

    @property
    def fps(self):
        try:
            self._return_data = self._output_queue.get(block=False)
        except queue.Empty:
            pass
        finally:
            return self._return_data.get(OutputVariables.FPS, None)

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: float):
        self._input_queue.put((InputVariables.TIMEOUT, timeout))
        self._timeout = timeout

    @property
    def poll_rate(self):
        return self._poll_rate

    @poll_rate.setter
    def poll_rate(self, poll_rate: float):
        self._input_queue.put((InputVariables.POLLRATE, poll_rate))
        self._poll_rate = poll_rate

    def read(self):
        return self._video_queue.get(block=False)

    def open(self, *args, **kwargs):
        self.release()
        success = self._capture.open(*args, **kwargs)
        if success:
            self.threaded_reader = threading.Thread(target=reader, args=(self._capture,
                                                                         self._video_queue,
                                                                         self._output_queue,
                                                                         self._input_queue,
                                                                         self._timeout,
                                                                         self._poll_rate), daemon=True)
            self.threaded_reader.start()
        return success

    def set(self, *args, **kwargs):
        return self._capture.set(*args, **kwargs)

    def get(self, propId: int):
        return self._capture.get(propId)

    def getBackendName(self):
        return self._capture.getBackendName()

    def isOpened(self):
        return self._capture.isOpened()

    def release(self):
        if self.threaded_reader:
            self._input_queue.put((InputVariables.QUIT,))
            logging.info("Waiting for threaded reader to quit.")
            self.threaded_reader.join()
        self._capture.release()
