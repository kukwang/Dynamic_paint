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
# parameters 1
# -----------------------------------------------------------------------------------------
cam_width, cam_height = 640, 480            # window size
max_distance_in_meters = 2                  # depth threshold
box_init = 20                               # half of initialize box length

frame_reduc = 100                           # Frame Reduction
smoothening = 7                             # smoothening cursor move
prev_x1, prev_y1 = 0, 0                     # previous cursor position
cur_x1, cur_y1 = 0, 0                       # current cursor position
velocity = 0                                # cursor velocity
distance = 0                                # distance between two points
past_time = 0                               # to calculate velocity and fps

# -----------------------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------------------
"""
# get color, depth image from depth camera
def get_frames(pl, al):
    # wait for a coherent pair of frames: depth and color
    frames = pl.wait_for_frames()

    # frames.get_depth_frame() is a 640x360 depth image
    # align the depth frame to color frame
    aligned_frames = al.process(frames)

    # get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        return False, False

    c_img = np.asanyarray(color_frame.get_data())
    d_img = np.asanyarray(aligned_depth_frame.get_data())
    c_img = cv2.flip(c_img, 1)
    d_img = cv2.flip(d_img, 1)

    return c_img, d_img
"""


# calculate distance between two points
def get_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (1 / 2)

# -----------------------------------------------------------------------------------------


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

config.enable_stream(rs.stream.depth, cam_width, cam_height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, cam_width, cam_height, rs.format.bgr8, 30)


# start streaming
profile = pipeline.start(config)
# -----------------------------------------------------------------------------------------

# getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# calculate distance that reflects camera's depth scale
max_distance = max_distance_in_meters / depth_scale

# create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# "align_to" is the stream type to which we plan to align depth frames
align_to = rs.stream.color
align = rs.align(align_to)

# assign HandDetector and Mouse class
detector = hand_tracing.HandDetector(maxHands=1)
mouse = mouse_control.Mouse()

while True:
    # -----------------------------------------------------------------------------------------
    # parameters 2
    # -----------------------------------------------------------------------------------------
    vib_init = 0            # vibration threshold
    vib_init_time = 0
    start_init = False      # start initialize flag
    finish_init = False     # initialization flag
    pressed_key = 0         # pressed key in keyboard
    velo_init_time = 0       # to calculate init velocity
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    # vibration threshold initialization
    # -----------------------------------------------------------------------------------------
    while not finish_init:
        # -----------------------------------------------------------------------------------------
        # get color, depth image from depth camera
        # -----------------------------------------------------------------------------------------
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(aligned_depth_frame.get_data())
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)
        # -----------------------------------------------------------------------------------------

        # find hand Landmarks
        color_img = detector.find_hands(color_img, draw=False)
        lmList, _ = detector.find_position(color_img, draw=False)

        # get the position of the index fingertip
        if len(lmList) != 0:
            # index fingertip position
            x1, y1 = lmList[8][1:]

            # check which finger is up
            fingers = detector.fingers_up()

            # left top: 0,0
            # rectangle: left top, right bottom
            left_top_x, left_top_y = cam_width // 2 - box_init, cam_height // 2 - box_init
            right_bottom_x, right_bottom_y = cam_width // 2 + box_init, cam_height // 2 + box_init
            cv2.rectangle(color_img, (cam_width // 2 - box_init, cam_height // 2 - box_init),
                          (cam_width // 2 + box_init, cam_height // 2 + box_init), (255, 0, 255), 2)

            # if index finger is up and fingertip is in the frame
            if fingers[1] and 0 <= x1 < 640 and 0 <= y1 < 480:

                cv2.circle(color_img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

                # if index finger is in the given area, start initialize
                if left_top_x < x1 < right_bottom_x and left_top_y < y1 < right_bottom_y:
                    # start measuring time
                    if not start_init:
                        vib_init_time = time.time()
                        # set init position
                        prev_x1, prev_y1 = x1, y1

                    start_init = True

                    cur_x1 = prev_x1 + (x1 - prev_x1) / smoothening
                    cur_y1 = prev_y1 + (y1 - prev_y1) / smoothening

                    # calculate threshold velocity
                    distance = get_distance([cur_x1, cur_y1], [prev_x1, prev_y1])
                    cur_time = time.time()
                    velocity = int(distance / (cur_time - velo_init_time) / 30)
                    velo_init_time = cur_time
                    prev_x1, prev_y1 = cur_x1, cur_y1

                    # if distance is large than vibration init, update it
                    if vib_init < velocity:
                        vib_init = velocity

                    # if init time is more than 3 second, finish vibration initialization
                    if cur_time - vib_init_time > 5:
                        finish_init = True
                        start_init = False

                # if index finger is out of given area, stop initialize
                else:
                    start_init = False
                    vib_init = 0

        # apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # dimension: 640x480 -> 640x480x3
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        # show fps and velocity
        cur_time = time.time()
        fps = 1 / (cur_time - past_time)
        past_time = cur_time
        cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(depth_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(color_img, str(velocity) + 'pixel / second / 30', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # display
        cv2.imshow("color_img", color_img)
        cv2.imshow("depth_img", depth_img)

        # ESC breaks the while loop
        pressed_key = cv2.waitKey(1)
        if pressed_key == 27:
            break

    if pressed_key == 27:
        break

    # -----------------------------------------------------------------------------------------
    # start
    # -----------------------------------------------------------------------------------------
    while True:
        # -----------------------------------------------------------------------------------------
        # get color, depth image from depth camera
        # -----------------------------------------------------------------------------------------
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
    
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
    
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
    
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
    
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(aligned_depth_frame.get_data())
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)
        # -----------------------------------------------------------------------------------------
    
        # 1. find hand Landmarks
        color_img = detector.find_hands(color_img, draw=True)
        lmList, _ = detector.find_position(color_img, draw=True)
    
        # 2. get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
    
            # 3. check which fingers are up
            fingers = detector.fingers_up()
            cv2.rectangle(color_img, (frame_reduc, frame_reduc), (cam_width - frame_reduc, cam_height - frame_reduc), (255, 0, 255), 2)
    
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
                        cur_x1 = np.interp(x1, (frame_reduc, cam_width - frame_reduc), (0, mouse_control.scr_width))
                        cur_y1 = np.interp(y1, (frame_reduc, cam_height - frame_reduc), (0, mouse_control.scr_height))
    
                        # 6. smoothen Values
                        if distance > 2:
                            cur_x1 = prev_x1 + (cur_x1 - prev_x1) / smoothening
                            cur_y1 = prev_y1 + (cur_y1 - prev_y1) / smoothening
    
                        # to remove effect of vibration
                        if velocity > vib_init:
                            # 7. move Mouse
                            # mouse.set_pos(0, 0): top right
                            # mouse.set_pos(scr_width, scr_height): bottom left
                            mouse.set_pos(cur_x1, cur_y1)
                            cv2.circle(color_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            cv2.circle(depth_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    
                        # calculate distance
                        distance = get_distance([cur_x1, cur_y1], [prev_x1, prev_y1])
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
                            if velocity > vib_init and length < 40:
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
        velocity = int(distance / (cur_time - past_time) / 30)
        past_time = cur_time
        cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(depth_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # show velocity of the mouse
        cv2.putText(color_img, str(velocity) + 'pixel / second / 30', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    
        # 12. display
        cv2.imshow("color_img", color_img)
        cv2.imshow("depth_img", depth_img)
    
        # ESC breaks the while loop
        pressed_key = cv2.waitKey(1)
        if pressed_key == 27:
            break

    if pressed_key == 27:
        break

# release camera and close the window
cv2.destroyAllWindows()
pipeline.stop()

