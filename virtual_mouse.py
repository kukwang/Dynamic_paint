"""
virtual_mouse.py
By: Murtaza Hassan
Youtube: https://www.youtube.com/watch?v=8gPONnGIPgw
Website: https://www.computervision.zone/courses/ai-virtual-mouse/
Modified by Kwangsoo Seol
Using win32api
"""
import pyrealsense2 as rs
import cv2
import numpy as np
import time

import hand_tracing
import mouse_control


# -----------------------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------------------
# calculate distance between two points
def cal_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** (1 / 2)


# -----------------------------------------------------------------------------------------
# parameters
# -----------------------------------------------------------------------------------------
wCam, hCam = 640, 480       # window size
frameR = 100                # Frame Reduction
vib_init = 0                # vibration threshold
smoothening = 7             # smoothening cursor move
prev_x1, prev_y1 = 0, 0     # previous cursor position
cur_x1, cur_y1 = 0, 0       # current cursor position
velocity = 0                # cursor velocity
distance = 0                # distance between two points (mainly previous - current cursor position)
past_time = 0                   # to calculate velocity

# -----------------------------------------------------------------------------------------
# connect to depth camera
# -----------------------------------------------------------------------------------------
# configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# try to find rgb camera
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
# if rgb camera is not found, exit the code
if not found_rgb:
    print("This code requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, wCam, hCam, rs.format.z16, 30)
config.enable_stream(rs.stream.color, wCam, hCam, rs.format.bgr8, 30)

# start streaming
profile = pipeline.start(config)
# -----------------------------------------------------------------------------------------

# getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# calculate max distance that reflects camera's depth scale
max_distance_in_meters = 2      # 2 meter
max_distance = max_distance_in_meters / depth_scale

# create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# "align_to" is the stream type to which we plan to align depth frames
align_to = rs.stream.color
align = rs.align(align_to)

# assign HandDetector and Mouse class
detector = hand_tracing.HandDetector(maxHands=1)
mouse = mouse_control.Mouse()

# -----------------------------------------------------------------------------------------
# vibration threshold initialization
# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
# start
# -----------------------------------------------------------------------------------------
while True:
    # -----------------------------------------------------------------------------------------
    # get color, depth image from depth camera
    # -----------------------------------------------------------------------------------------
    # wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    # frames.get_depth_frame() is a 640x360 depth image
    # align the depth frame to color frame
    aligned_frames = align.process(frames)

    # get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(aligned_depth_frame.get_data())
    color_img = cv2.flip(color_img, 1)
    depth_img = cv2.flip(depth_img, 1)
    # -----------------------------------------------------------------------------------------

    # 1. find hand Landmarks
    color_img = detector.find_hands(color_img, draw=True)
    lmList, bbox = detector.find_position(color_img, draw=True)

    # 2. get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(color_img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # to prevent indexError
        if 0 <= x1 < 640 and 0 <= y1 < 480:
            # get distance from depth sensor to fingertip
            index_distance = depth_img[y1][x1]

            # control the mouse only when it is closer than max_distance
            if index_distance <= max_distance:
                # 4. only Index Finger : Moving Mode
                # index finger is stretched and distance <= max distance, middle finger is bent
                if fingers[1] == 1 and fingers[2] == 0:
                    # 5. convert Coordinates
                    cur_x1 = np.interp(x1, (frameR, wCam - frameR), (0, mouse_control.wScr))
                    cur_y1 = np.interp(y1, (frameR, hCam - frameR), (0, mouse_control.hScr))

                    # 6. smoothen Values
                    if distance > 2:
                        cur_x1 = prev_x1 + (cur_x1 - prev_x1) / smoothening
                        cur_y1 = prev_y1 + (cur_y1 - prev_y1) / smoothening

                    # to remove effect of vibration
                    if velocity > 3:
                        # 7. move Mouse
                        # mouse.set_pos(0, 0): top right
                        # mouse.set_pos(wScr, hScr): bottom left
                        mouse.set_pos(cur_x1, cur_y1)
                        cv2.circle(color_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        cv2.circle(depth_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

                    # calculate distance
                    distance = cal_distance([cur_x1, cur_y1], [prev_x1, prev_y1])
                    prev_x1, prev_y1 = cur_x1, cur_y1

                # to prevent indexError
                if 0 <= x2 < 640 and 0 <= y2 < 480:
                    # 8. both Index and middle fingers are up : Clicking Mode
                    # index finger and middle finger are stretched and distance <= max distance
                    if fingers[1] == 1 and fingers[2] == 1:

                        # 9. find distance between fingers
                        length, color_img, lineInfo = detector.find_distance(8, 12, color_img)
                        _, depth_img, _ = detector.find_distance(8, 12, depth_img)

                        # 10. click mouse if distance short
                        # length < 40:
                        if velocity > 3 and length < 40:
                            cv2.circle(color_img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            cv2.circle(depth_img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            mouse.left_click()

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # dimension: 640x480 -> 640x480x3
    depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    # 11. fps and velocity(pixel / sec / 50)
    cur_time = time.time()
    fps = 1 / (cur_time - past_time)
    velocity = int(distance / (cur_time - past_time) / 50)
    past_time = cur_time
    cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(depth_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # -----------------------------------------------------------------------------------------
    # show velocity of the mouse
    # -----------------------------------------------------------------------------------------
    cv2.putText(color_img, str(velocity) + 'pixel / second / 50', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # 12. display
    cv2.imshow("color_img", color_img)
    cv2.imshow("depth_img", depth_img)

    # ESC breaks the while loop
    if cv2.waitKey(1) == 27:
        break

# -----------------------------------------------------------------------------------------
# release camera and close the window
cv2.destroyAllWindows()
pipeline.stop()
