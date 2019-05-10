#!/user/bin/python
# -*- coding: utf-8

import cv2
import numpy as np

class Chroma:
    def __init__(self, name="", cal_f_max=100, hard_set=False):
        self.name = name

        self.cal_f_cur = 0
        self.cal_f_max = cal_f_max

        self.hard_set = hard_set

        self.Hmin, self.Hmax = 3, 18
        self.Hmean, self.Hstdv = 0, 0
        self.Hmean_ac, self.Hstdv_ac = 0, 0

        self.Smin, self.Smax = 60, 255
        self.Smean, self.Sstdv = 0, 0
        self.Smean_ac, self.Sstdv_ac = 0, 0
        self.lower_bound, self.upper_bound = 0, 0

        self.hue_max_m = 15
        self.hue_min_m = 3

        self.sat_max_m = 255
        self.sat_min_m = 50
        
    def init(self, cal_f_max=100):
        self.cal_f_cur = 0
        self.cal_f_max = cal_f_max

        self.Hmin, self.Hmax = 0, 0
        self.Hmean, self.Hstdv = 0, 0
        self.Hmean_ac, self.Hstdv_ac = 0, 0

        self.Smin, self.Smax = 0, 0
        self.Smean, self.Sstdv = 0, 0
        self.Smean_ac, self.Sstdv_ac = 0, 0
        self.lower_bound, self.upper_bound = 0, 0
        return True

    def calibrate(self, image):
        if self.cal_f_cur > self.cal_f_max:
            return False

        if self.cal_f_cur < self.cal_f_max:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_c, s_c, v_c = cv2.split(hsv)

            Hmean_buffer, Hstdv_buffer = cv2.meanStdDev(h_c)
            Smean_buffer, Sstdv_buffer = cv2.meanStdDev(s_c)

            self.Hmean_ac += Hmean_buffer
            self.Hstdv_ac += Hstdv_buffer

            self.Smean_ac += Smean_buffer
            self.Sstdv_ac += Sstdv_buffer

            print(self.cal_f_cur)
        elif self.cal_f_cur is self.cal_f_max:
            self.Hmean = self.Hmean_ac / self.cal_f_max
            self.Hstdv = self.Hstdv_ac / self.cal_f_max

            self.Smean = self.Smean_ac / self.cal_f_max
            self.Sstdv = self.Sstdv_ac / self.cal_f_max
            print("done")
        self.cal_f_cur += 1
        return True

    def apply(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.hard_set is False:
            self.Hmin = np.clip(self.Hmean - (self.hue_min_m * self.Hstdv), 0, 255)
            self.Hmax = np.clip(self.Hmean + (self.hue_max_m * self.Hstdv), 0, 255)
            self.Smin = np.clip(self.Smean - (self.sat_min_m * self.Sstdv), 0, 255)
            self.Smax = np.clip(self.Smean + (self.sat_max_m * self.Sstdv), 0, 255)

        self.lower_bound = np.array([self.Hmin, self.Smin, 0], dtype=np.uint8)
        self.upper_bound = np.array([self.Hmax, self.Smax, 255], dtype=np.uint8)
        inv_mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        mask = 255 - inv_mask
        return mask, inv_mask

    def set_hue_max_m(self, x):
        self.hue_max_m = x / 100

    def set_hue_min_m(self, x):
        self.hue_min_m = x / 100

    def set_sat_max_m(self, x):
        self.sat_max_m = x / 100

    def set_sat_min_m(self, x):
        self.sat_min_m = x / 100

    # parametros para calibración manual
    def set_hue_max(self, x):
        self.Hmax = x

    def set_hue_min(self, x):
        self.Hmin = x

    def set_sat_max(self, x):
        self.Smax = x

    def set_sat_min(self, x):
        self.Smin = x

    def createCalibrationWindow(self):
        windowName = self.name + " Calibration"
        cv2.namedWindow(windowName)

        if(self.hard_set is not True):
            cv2.createTrackbar("H Max", windowName, 100, 2000, self.set_hue_max_m)
            cv2.createTrackbar("H Min", windowName, 100, 2000, self.set_hue_min_m)
            cv2.createTrackbar("S Max", windowName, 100, 2000, self.set_sat_max_m)
            cv2.createTrackbar("S Min", windowName, 100, 2000, self.set_sat_min_m)
        else:
            cv2.createTrackbar("H Max", windowName, 15, 255, self.set_hue_max)
            cv2.createTrackbar("H Min", windowName, 3, 255, self.set_hue_min)
            cv2.createTrackbar("S Max", windowName, 255, 255, self.set_sat_max)
            cv2.createTrackbar("S Min", windowName, 58, 255, self.set_sat_min)

    def destroyCalibrationWindow(self):
        windowName = self.name + " Calibration"
        cv2.destroyWindow(windowName)

    
