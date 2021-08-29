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
import painter_class

# -----------------------------------------------------------------------------------------
# parameters
# -----------------------------------------------------------------------------------------
cam_width, cam_height = 640, 480    # window size
max_distance_in_meters = 1          # depth threshold
base_depth_img = []                 # base depth image info

frame_reduc = 100                   # Frame Reduction
smoothening = 7                     # smoothening cursor move
prev_x1, prev_y1 = 0, 0             # previous cursor position
cur_x1, cur_y1 = 0, 0               # current cursor position
stopped_x1, stopped_y1 = 0, 0       # last update cursor position
virtual_distance = 0                # distance between last update position and current position

vib_dis_init = 4                    # vibration distance threshold
pressed_key = 0                     # pressed key in keyboard
click_height = 5                    # verify click operation
velocity = 0                        # cursor velocity
distance = 0                        # distance between two points
past_time = 0                       # to calculate velocity and fps


# -----------------------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------------------
# calculate distance between two points
def get_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (1 / 2)


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
paint = painter_class.Paint()
# -----------------------------------------------------------------------------------------
while True:
    # -----------------------------------------------------------------------------------------
    # base initialization
    # -----------------------------------------------------------------------------------------
    while True:
        # wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image

        # validate that both frames are valid
        if not aligned_depth_frame:
            continue

        else:
            base_depth_img = np.asanyarray(aligned_depth_frame.get_data())
            base_depth_img = cv2.flip(base_depth_img, -1)
            break

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
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(aligned_depth_frame.get_data())
        color_img = cv2.flip(color_img, -1)
        depth_img = cv2.flip(depth_img, -1)
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
                    cv2.circle(depth_img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                    stopped_x1, stopped_y1 = prev_x1, prev_y1

                # calculate distance
                distance = get_distance([cur_x1, cur_y1], [prev_x1, prev_y1])
                prev_x1, prev_y1 = cur_x1, cur_y1
                # click part
                # get distance from depth sensor to fingertip
                index_distance = depth_img[y1][x1]
                print('index dis:', index_distance, 'base dis:', base_depth_img[y1][x1])
                #if index_distance > base_depth_img[y1][x1] + click_height:
                #    mouse.left_unpress()
                #elif index_distance <= base_depth_img[y1][x1] + click_height:
                #    mouse.left_press()

        # apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # dimension: 640x480 -> 640x480x3
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        # fps and velocity(pixel / sec / 50)
        cur_time = time.time()
        fps = 1 / (cur_time - past_time)
        velocity = int(distance / (cur_time - past_time) / 50)
        past_time = cur_time
        cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(depth_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(color_img, str(velocity) + 'pixel / second / scale_factor', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(color_img, str(virtual_distance) + 'pixel', (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # display
        cv2.imshow("color_img", color_img)
        cv2.imshow("depth_img", depth_img)

        # ESC breaks the while loop twice
        # 'r' key breaks the while loop and restart base initialization(114 is 'r' in ascii code)
        pressed_key = cv2.waitKey(1)
        if pressed_key == 27 or pressed_key == 114:
            break
    if pressed_key == 27:
        break

# release camera and close the window
cv2.destroyAllWindows()
pipeline.stop()
