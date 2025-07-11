import cv2 as cv
import numpy as np

capture = cv.VideoCapture('Ball_Tracking.mp4')
l_boundary = np.array([30, 120, 30])
h_boundary = np.array([90, 255, 255])
while True:
    isTrue, frame = capture.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    target = cv.inRange(hsv, l_boundary, h_boundary)
    target = cv.erode(target, None, iterations=2)
    target = cv.dilate(target, None, iterations=2)

    contours, _ = cv.findContours(target, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for contour in contours:

        if cv.contourArea(contour) < 500:
            continue

        ((x, y), radius) = cv.minEnclosingCircle(contour)

        if radius > 10:
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv.imshow('ball_tracing', frame)
    if cv.waitKey(36) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
