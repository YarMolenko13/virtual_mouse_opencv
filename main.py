from cv2 import cv2
import numpy as np
import HandDetectorModule as hdm
import time
import autopy

CAM_WIDTH, CAM_HEIGHT = 680, 480
SCREEN_WIDTH, SCREEN_HEIGHT = autopy.screen.size()
FRAME_REDUCTION = 90
SMOOTH = 7

p_time = 0

prev_loc_x, prev_loc_y = 0, 0
current_loc_x, current_loc_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

detector = hdm.handDetector(maxHands=1)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame, draw=False)
    el_list, b_box = detector.findPosition(frame, draw=False)

    cv2.rectangle(frame, (FRAME_REDUCTION, FRAME_REDUCTION),
                  (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION),
                  (255, 0, 255), 2)

    if len(el_list) != 0:
        x1, y1 = el_list[8][1:]
        x2, y2 = el_list[12][1:]

        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, SCREEN_WIDTH))
            y3 = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, SCREEN_HEIGHT))

            current_loc_x = prev_loc_x + (x3 - prev_loc_x) / SMOOTH
            current_loc_y = prev_loc_y + (y3 - prev_loc_y) / SMOOTH

            autopy.mouse.move(current_loc_x, current_loc_y)
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            prev_loc_x, prev_loc_y = current_loc_x, current_loc_y

        if fingers[1] == 1 and fingers[2] == 1:
            length, frame, line_info = detector.findDistance(8, 12, frame)

            if length < 40:
                cv2.circle(frame, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    frame = cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 3)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
