"""
virtual_mouse.py
By: Murtaza Hassan
Youtube: https://www.youtube.com/watch?v=8gPONnGIPgw
Website: https://www.computervision.zone/courses/ai-virtual-mouse/
Modified by Kwangsoo Seol
Using win32api
"""
import cv2
import numpy as np
import time

import hand_tracing
import mouse_control
import painter_class


# -----------------------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------------------
# calculate distance between two points
def get_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (1 / 2)


# -----------------------------------------------------------------------------------------
# parameters
# -----------------------------------------------------------------------------------------
cam_width, cam_height = 640, 480    # window size

frame_reduc = 100                   # Frame Reduction
smoothening = 7                     # smoothening cursor move
prev_x1, prev_y1 = 0, 0             # previous cursor position
cur_x1, cur_y1 = 0, 0               # current cursor position
stopped_x1, stopped_y1 = 0, 0       # last update cursor position
virtual_distance = 0                # distance between last update position and current position

vib_dis_init = 4                    # vibration distance threshold
pressed_key = 0                     # pressed key in keyboard
velocity = 0                        # cursor velocity
distance = 0                        # distance between two points
past_time = 0                       # to calculate velocity and fps

# assign HandDetector and Mouse class
detector = hand_tracing.HandDetector(maxHands=1)
mouse = mouse_control.Mouse()
paint = painter_class.Paint()
# -----------------------------------------------------------------------------------------

# image size: 640 x 480
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# -----------------------------------------------------------------------------------------
# start
# -----------------------------------------------------------------------------------------
while True:
    # -----------------------------------------------------------------------------------------
    # get color image from camera
    # -----------------------------------------------------------------------------------------
    _, color_img = capture.read()
    color_img = cv2.flip(color_img, -1)
    # -----------------------------------------------------------------------------------------

    # find hand Landmarks
    color_img = detector.find_hands(color_img, draw=True)
    lmList, _ = detector.find_position(color_img, draw=False)

    # get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]

        # check which fingers are stretched
        cv2.rectangle(color_img, (frame_reduc, frame_reduc), (cam_width - frame_reduc, cam_height - frame_reduc),
                      (255, 0, 255), 2)

        # to prevent indexError
        if 0 <= x1 < 640 and 0 <= y1 < 480:
            # only Index Finger : Moving Mode
            # index finger is stretched and distance <= max distance, middle finger is bent
            # convert Coordinates
            cur_x1 = np.interp(x1, (frame_reduc, cam_width - frame_reduc), (0, mouse_control.scr_width))
            cur_y1 = np.interp(y1, (frame_reduc, cam_height - frame_reduc), (0, mouse_control.scr_height))

            # smoothen Values
            cur_x1 = int(prev_x1 + (cur_x1 - prev_x1) / smoothening)
            cur_y1 = int(prev_y1 + (cur_y1 - prev_y1) / smoothening)

            virtual_distance = get_distance([cur_x1, cur_y1], [stopped_x1, stopped_y1])
            # to reduce effect of vibration
            if virtual_distance > vib_dis_init:
                # move Mouse
                # mouse.set_pos(0, 0): top right
                # mouse.set_pos(scr_width, scr_height): bottom left
                #mouse.set_pos(cur_x1, cur_y1)
                # // 2: paint area is quarter of the screen size
                paint.draw(mouse.clicked, [prev_x1//2, prev_y1//2], [cur_x1//2, cur_y1//2], velocity, False)
                cv2.circle(color_img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                stopped_x1, stopped_y1 = prev_x1, prev_y1

            # calculate distance
            distance = get_distance([cur_x1, cur_y1], [prev_x1, prev_y1])
            prev_x1, prev_y1 = cur_x1, cur_y1
            # click part

    # fps and velocity(pixel / sec / 50)
    cur_time = time.time()
    fps = 1 / (cur_time - past_time)
    velocity = int(distance / (cur_time - past_time) / 50)
    past_time = cur_time
    cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(color_img, str(velocity) + 'pixel / second / scale_factor', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    cv2.putText(color_img, str(virtual_distance) + 'pixel', (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # display
    cv2.imshow("color_img", color_img)

    # ESC breaks the while loop twice
    pressed_key = cv2.waitKey(1)
    if pressed_key == 27:
        break

# release camera and close the window
capture.release()
cv2.destroyAllWindows()
