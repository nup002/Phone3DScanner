# -*- coding: utf-8 -*-
"""
Displays a checkerboard with a known size in physical dimensions (mm) on the main display.
@author: magne.lauritzen
"""

from screeninfo import get_monitors
import numpy as np
import cv2
CHECKERBOARD_SHAPE = (6, 9)  # Width, height

def build_checkerboard(nw, nh, pxw=1, pxh=1):
    arr = np.ones(shape=(nh*pxh, nw*pxw), dtype=bool)
    for check_w in np.arange(nw):
        arr[:, check_w*pxw:] = ~arr[:, check_w*pxw:]
    for check_h in np.arange(nh):
        arr[check_h*pxh:] = ~arr[check_h*pxh:]
    return arr

monitor = None
for m in get_monitors():
    if m.is_primary:
        monitor = m
dpm_w = monitor.width / monitor.width_mm
dpm_h = monitor.height / monitor.height_mm

base_pixel_size = np.array([CHECKERBOARD_SHAPE[0]*dpm_w, CHECKERBOARD_SHAPE[1]*dpm_h])
factor = min((0.8*monitor.width/base_pixel_size[0], 0.8*monitor.height/base_pixel_size[1]))
final_pixel_size = (base_pixel_size*factor).astype(int)
final_mm_size = final_pixel_size/np.array([dpm_w, dpm_h])

print("-------------------------------")
print(f"Displaying checkerboard of size {final_mm_size[0]:.2f}mm x {final_mm_size[1]:.2f}mm.")
print(f"Using monitor \"{monitor.name}\". \n\tResolution: width={monitor.width}px, height={monitor.height}px\n\t"
      f"Dimension : width={monitor.width_mm}mm, height={monitor.height_mm}mm.")
print("-------------------------------")
checkerboard = build_checkerboard(*CHECKERBOARD_SHAPE).astype(float)
imS = cv2.resize(checkerboard, final_pixel_size, interpolation=cv2.INTER_NEAREST)
while True:
    cv2.imshow("checkerboard", imS)
    if cv2.waitKey(1) == ord("q"):
        break
