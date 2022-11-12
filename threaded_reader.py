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


# noinspection PyArgumentList
class InputVariables(Enum):
    TIMEOUT = auto()
    POLLRATE = auto()
    QUIT = auto()

# noinspection PyArgumentList
class OutputVariables(Enum):
    POLLRATE = auto()
    FPS = auto()


def queue_put(q: queue.Queue, data: Any):
    """ Places an item in a LifoQueue. If the queue is full, it removes one element to make room. """
    if q.full():
        q.get(block=False)
    q.put(data, block=False)


def sleep_and_poll_queues(delay):
    """
    Function to provide accurate time delay
    """
    _ = time.perf_counter() + delay
    while time.perf_counter() < _:
        pass


def reader(capture: cv2.VideoCapture, video_queue: queue.Queue, out_queue: queue.Queue,
           in_queue: queue.Queue, timeout: float, poll_rate: float):
    def sleep_and_poll_queues(delay):
        """
        Function to provide accurate time delay while polling queues for new data.
        """
        _ = time.perf_counter() + delay
        while time.perf_counter() < _:
            pass

    poll_period_deque = deque()
    frame_period_deque = deque()
    if capture.isOpened():
        quitflag = False
        poll_period = 1 / poll_rate
        time_since_frame = 0
        last_emit_timestamp = 0
        last_settings_poll_timestamp = 0
        prev_frame_timestamp = time.time()
        prev_read_timestamp = time.perf_counter()
        while time_since_frame < timeout and not quitflag:
            # Get time since last call and sleep if needed to reach poll_rate
            time_since_last_read = time.perf_counter() - prev_read_timestamp
            sleep_until = time.perf_counter() + max(0., poll_period - time_since_last_read)

            # Poll new settings at 10hz while sleeping
            while True:
                now = time.perf_counter()
                if now - last_settings_poll_timestamp > 0.1:
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
                            quitflag = True
                            break
                    except queue.Empty:
                        pass
                    last_settings_poll_timestamp = now

                if time.perf_counter() > sleep_until:
                    break

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

            # Emit statistics every 1 second
            if prev_read_timestamp - last_emit_timestamp > 1:
                mean_poll_period = mean(poll_period_deque)
                poll_period_deque.clear()
                mean_frame_period = mean(frame_period_deque)
                frame_period_deque.clear()
                queue_put(out_queue, {OutputVariables.POLLRATE: 1 / mean_poll_period,
                                      OutputVariables.FPS: 1 / mean_frame_period})
                last_emit_timestamp = prev_read_timestamp

    queue_put(video_queue, (None, None))


class ThreadedVideoCapture:
    """
    This is a class that can be used in place of cv2.VideoCapture. It performs frame reads in a separate thread, which
    frees up the main thread to do other tasks.

    ThreadedVideoCapture can be used in the same way as VideoCapture. It takes a few extra keyword arguments.

    ThreadedVideoCapture can be used as a context manager. If you do not use it as a context manager, you must
    ensure to call release() when you are done with it.
    """
    def __init__(self, *args, timeout: float = 1, poll_rate: float = 100, frame_queue_size: int = 1,
                 **kwargs):
        self._capture = cv2.VideoCapture()
        self._video_queue = queue.Queue(maxsize=frame_queue_size)
        self._output_queue = queue.Queue(maxsize=1)
        self._input_queue = queue.Queue()
        self._timeout = timeout
        self._poll_rate = poll_rate
        self._return_data = {}
        self.threaded_reader: Optional[threading.Thread] = None
        self.open(*args, **kwargs)

    def read(self):
        """
        Returns one frame from the video queue. If the queue is empty, returns None.

        Returns
        -------
        ret     : Optional bool. True if a frame is available. If the threaded reader has quit, it is None.
        frame   : np.ndarray or None.
        """
        try:
            return self._video_queue.get(block=False)
        except queue.Empty:
            return False, None

    def open(self, *args, **kwargs):
        """
        Opens a video stream. Wraps VideoCapture.open and takes identical arguments.

        Parameters
        ----------
        args    : Arguments for VideoCapture.open
        kwargs  : Keyword arguments for VideoCapture.open

        Returns
        -------
        success : bool. Whether the source was successfully opened.
        """
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

    def set(self, propId: int, value: int):
        """ Wrapper for VideoCapture.set. """
        return self._capture.set(propId, value)

    def get(self, propId: int):
        """ Wrapper for VideoCapture.get. """
        return self._capture.get(propId)

    def getBackendName(self):
        """ Wrapper for VideoCapture.getBackendName. """
        return self._capture.getBackendName()

    def isOpened(self):
        """ Wrapper for VideoCapture.isOpened. """
        return self._capture.isOpened()

    def release(self):
        """
        Stops the reader thread and releases the VideoCapture.
        """
        if self.threaded_reader:
            self._input_queue.put((InputVariables.QUIT,))
            logging.info("Waiting for threaded reader to quit.")
            self.threaded_reader.join()
        self._capture.release()

    @property
    def actual_poll_rate(self):
        """ Returns the most recent actual poll rate reported by the reader thread. The poll rate value is updated
        once per second."""
        try:
            self._return_data = self._output_queue.get(block=False)
        except queue.Empty:
            pass
        finally:
            return self._return_data.get(OutputVariables.POLLRATE, None)

    @property
    def fps(self):
        """ Returns the most recent frames per second reported by the reader thread. The FPS value is updated once per
        second."""
        try:
            self._return_data = self._output_queue.get(block=False)
        except queue.Empty:
            pass
        finally:
            return self._return_data.get(OutputVariables.FPS, None)

    @property
    def timeout(self):
        """ Return the reader thread timeout value. """
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: float):
        """
        Set the reader thread timeout. If no new frames have been received from the source (camera, file, stream,
        etc), the reader thread will quit.

        Parameters
        ----------
        timeout : Timeout value in seconds.
        """
        self._input_queue.put((InputVariables.TIMEOUT, timeout))
        self._timeout = timeout

    @property
    def poll_rate(self):
        """ Returns the reader thread poll rate. """
        return self._poll_rate

    @poll_rate.setter
    def poll_rate(self, poll_rate: float):
        """
        Set the reader thread poll rate. This value is the highest rate (in calls per second) to capture.grab() that
        the reader thread is allowed to make.

        Parameters
        ----------
        poll_rate   : Poll rate in calls per second
        """
        self._input_queue.put((InputVariables.POLLRATE, poll_rate))
        self._poll_rate = poll_rate

    def __enter__(self):
        return self

    def __exit__(self):
        self.release()
