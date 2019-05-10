import cv2
import numpy as np

def detect_hand(roi, mass_thresh, state_thresh=-1):
    kernel = np.ones((5, 5), dtype=np.uint8)
    image = roi.copy()
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    contornos = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[1]

    if len(contornos) > 0:
        contorno = max(contornos, key=lambda x: cv2.moments(x)['m00'])
        M = cv2.moments(contorno)
        A = M["m00"]
        if A < mass_thresh:
            return "empty"

        P = cv2.arcLength(contorno, True)
        compactness = P**2 / A
        if state_thresh is -1:
            return "found"

        if compactness > state_thresh:
            return "open"
        else:
            return "closed"

def render(target, image, image_mask=None, x=0, y=0, width=640, height=480):
    output = target.copy()
    inp = image.copy()
    try:
        output[y:y+height, x:x+width] = cv2.subtract(output[y:y+height, x:x+width], output[y:y+height, x:x+width], mask = image_mask)
        output[y:y+height, x:x+width] = cv2.add(output[y:y+height, x:x+width], inp, mask = image_mask)
    except:
        print("couldnt render")
    return output

def max_contour(roi, linea=-1, kernel=-1, min_area=-1):
    width, height = roi.shape
    r = np.zeros((width, height), dtype=np.uint8)
    if kernel is -1:
        kernel = np.ones((5, 5), dtype=np.uint8)
    image = roi.copy()
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    contornos = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(contornos) > 0:
        contorno = max(contornos, key=lambda x: cv2.moments(x)['m00'])
        if min_area is -1 or cv2.moments(contorno)['m00'] > min_area:
            cv2.drawContours(r, [contorno], 0, (255, 255, 255), linea)
            return r
    return r

def get_centroid(roi):
    ok = True
    M = cv2.moments(roi)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        cX = None
        cY = None
        ok = False
    return cX, cY, ok
