import numpy as np
import cv2

calibration_state = 0
debug_state = 1
running_state = 2
state = calibration_state

calibration_frame_max = 100
calibration_frame_current = 0
Hmin, Hmax, Hmean, Hstdv = 0, 0, 0, 0
Smin, Smax, Smean, Sstdv = 0, 0, 0, 0
lower_bound, upper_bound = 0, 0

beta_1 = 2.5
beta_2 = 2.5

chroma_mask = 0

def initialize_calibration():
    print("restarting calibration")
    calibration_frame_max = 100
    calibration_frame_current = 0
    Hmin, Hmax, Hmean, Hstdv = 0, 0, 0, 0
    Smin, Smax, Smean, Sstdv = 0, 0, 0, 0
    state = calibration_state
    
def calculate_bounds():
    Hmin = np.clip(Hmean - ((beta_1/100) * Hstdv), 0, 255)
    Hmax = np.clip(Hmean + ((beta_1/100) * Hstdv), 0, 255)
    Smin = np.clip(Smean - ((beta_2/100) * Sstdv), 0, 255)
    Smax = np.clip(Smean + ((beta_2/100) * Sstdv), 0, 255)
    lower_bound = np.array([Hmin, Smin, 50], dtype=np.uint8)
    upper_bound = np.array([Hmax, Smax, 255], dtype=np.uint8)
    chroma_mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

def change_b1(x):
    print("beta 1:", x)
    print(Hmin, Hmax, Hmean, Hstdv)
    beta_1 = x

def change_b2(x):
    print("beta 2:", x)
    print(Smin, Smax, Smean, Sstdv)
    beta_2 = x

cv2.namedWindow("Sliders")
cv2.createTrackbar("Beta 1", "Sliders", 6, 10, change_b1)
cv2.createTrackbar("Beta 2", "Sliders", 6, 10, change_b2)

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if state is calibration_state:
        print("Current calibration frame:", calibration_frame_current)
        #split hsv channels
        h, s, v = cv2.split(frame_hsv)

        #calculate mean and stdv for current frames h and s channels
        buffer_Hmean, buffer_Hstdv = cv2.meanStdDev(h)
        buffer_Smean, buffer_Sstdv = cv2.meanStdDev(s)
        
        #accumulate the buffers
        Hmean += buffer_Hmean
        Hstdv += buffer_Hstdv
        Smean += buffer_Smean
        Sstdv += buffer_Sstdv

        calibration_frame_current += 1
        if calibration_frame_current is calibration_frame_max - 1:
            #calibration algorithm
            Hmean = Hmean / calibration_frame_max
            Hstdv = Hstdv / calibration_frame_max
            Smean = Smean / calibration_frame_max
            Sstdv = Sstdv / calibration_frame_max
    
            Hmin = np.clip(Hmean - (beta_1 * Hstdv), 0, 255)
            Hmax = np.clip(Hmean + (beta_1 * Hstdv), 0, 255)
            Smin = np.clip(Smean - (beta_2 * Sstdv), 0, 255)
            Smax = np.clip(Smean + (beta_2 * Sstdv), 0, 255)
            lower_bound = np.array([Hmin, Smin, 0], dtype=np.uint8)
            upper_bound = np.array([Hmax, Smax, 255], dtype=np.uint8)
            chroma_mask = 255 - cv2.inRange(frame_hsv, lower_bound, upper_bound)
            kernel = np.ones((3,3), np.uint8)
            chroma_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_OPEN, kernel)

            #next state change
            state = debug_state
            print("Hmean:", Hmean, "Hstdv:", Hstdv, "Hmin:", Hmin, "Hmax:", Hmax)
            print("Smean:", Smean, "Sstdv:", Sstdv, "Smin:", Smin, "Smax:", Smax)
            print("going to debug state")

    elif state is debug_state:
            Hmin = np.clip(Hmean - (beta_1 * Hstdv), 0, 255)
            Hmax = np.clip(Hmean + (beta_1 * Hstdv), 0, 255)
            Smin = np.clip(Smean - (beta_2 * Sstdv), 0, 255)
            Smax = np.clip(Smean + (beta_2 * Sstdv), 0, 255)
            lower_bound = np.array([Hmin, Smin, 0], dtype=np.uint8)
            upper_bound = np.array([Hmax, Smax, 255], dtype=np.uint8)
            chroma_mask = 255 - cv2.inRange(frame_hsv, lower_bound, upper_bound)
            kernel = np.ones((3,3), np.uint8)
            chroma_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_OPEN, kernel)
            #chroma_mask = cv2.erode(chroma_mask, kernel, iterations = 1)
#    elif state is running_state:

    if state is calibration_state:
        cv2.imshow("asdf", frame)
    elif state is debug_state:
        calibrated_frame = cv2.bitwise_and(frame, frame, mask=chroma_mask)
        cv2.imshow("asdf", calibrated_frame)
        cv2.imshow("mask", chroma_mask)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        #restarting calibration
        print("restarting calibration")
        calibration_frame_max = 100
        calibration_frame_current = 0
        Hmin, Hmax, Hmean, Hstdv = 0, 0, 0, 0
        Smin, Smax, Smean, Sstdv = 0, 0, 0, 0
        state = calibration_state
        #initialize_calibration()

    # Quit the thing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()