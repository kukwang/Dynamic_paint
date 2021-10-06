"""
virtual_mouse.py
By: Murtaza Hassan
Youtube: https://www.youtube.com/watch?v=8gPONnGIPgw
Website: https://www.computervision.zone/courses/ai-virtual-mouse/
Modified by Kwangsoo Seol
"""
import pyrealsense2 as rs
import cv2
import numpy as np
import time

import hand_tracing
import painter_class
from win32api import GetSystemMetrics


# -----------------------------------------------------------------------------------------
# helper function
# -----------------------------------------------------------------------------------------
# calculate distance between two points
def get_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (1 / 2)


# calculate interpolated position
def get_interp_pos(pt, fr, cw, ch, scw, sch):
    interp_pt = [int(np.interp(pt[0], (fr, cw - fr), (0, scw))), int(np.interp(pt[1], (fr, ch - fr), (0, sch)))]
    return interp_pt


# calculate smoothened position
def get_smoothen_pos(prev_pt, cur_pt, sm):
    sm_pt = [int(prev_pt[0] + (cur_pt[0] - prev_pt[0]) / sm), int(prev_pt[1] + (cur_pt[1] - prev_pt[1]) / sm)]
    return sm_pt


# -----------------------------------------------------------------------------------------
# parameters
# -----------------------------------------------------------------------------------------
cam_width, cam_height = 640, 480    # window size
max_dis_in_meters = 1.5             # maximum sensing distance in meters

frame_reduc = 100                   # Frame Reduction
smoothening = 7                     # smoothening cursor move
velocity = 0                        # cursor velocity
distance = 0                        # distance between two points
past_time = 0                       # to calculate velocity and fps
prev_index = [0, 0]                 # previous cursor position
cur_index = [0, 0]                  # current cursor position
stopped_index = [0, 0]              # last updated cursor position

scr_width, scr_height = GetSystemMetrics(0), GetSystemMetrics(1)    # get width and height of the monitor

vib_dis = 4                         # vibration distance threshold
in_mid_max_dis = 200                # maximum sensing distance between index and middle fingertip
pressed_key = 0                     # pressed key in keyboard
is_initial = True                   # to reset prev, cur position

prev_depth, cur_depth = 0, 0

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

# getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# calculate distance that reflects camera's depth scale
max_dis = max_dis_in_meters / depth_scale

# create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# "align_to" is the stream type to which we plan to align depth frames
align_to = rs.stream.color
align = rs.align(align_to)

# assign HandDetector and Mouse class
detector = hand_tracing.HandDetector(maxHands=1)
paint = painter_class.Paint(width=int(scr_width), height=int(scr_height))

# -----------------------------------------------------------------------------------------
# start drawing
# -----------------------------------------------------------------------------------------
while True:
    # -----------------------------------------------------------------------------------------
    # get color, depth image from depth camera
    # -----------------------------------------------------------------------------------------
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # if one of two(color, depth image) is not detected, continue
    if not aligned_depth_frame or not color_frame:
        continue

    # data type of color_img: numpy.uint8(cv2.CV_8U)
    # data type of depth_img: numpy.uint16(cv2.CV_16U)
    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(aligned_depth_frame.get_data())
    color_img = cv2.flip(color_img, 1)
    depth_img = cv2.flip(depth_img, 1)

    color_img_reduc = color_img[frame_reduc:cam_height - frame_reduc, frame_reduc:cam_width - frame_reduc]
    depth_img_reduc = depth_img[frame_reduc:cam_height - frame_reduc, frame_reduc:cam_width - frame_reduc]
    # -----------------------------------------------------------------------------------------

    # find hand and its landmarks
    color_img = detector.find_hands(color_img, draw=True)
    lmList, _ = detector.find_position(color_img, draw=False)

    # if hand is detected, proceed to the next process
    if len(lmList) != 0:
        # get the fingertip of the index and middle fingers
        index_pos = lmList[8][1:]
        middle_pos = lmList[12][1:]

        # check if fingers are up
        fingers = detector.fingers_up()
        # show fingertip sense area in the color image
        cv2.rectangle(color_img, (frame_reduc, frame_reduc),
                      (cam_width - frame_reduc, cam_height - frame_reduc), (255, 0, 255), 2)

        # to prevent indexError, limit the position of index fingertip point
        # sometimes index fingertip position
        if 0 <= index_pos[0] < 640 and 0 <= index_pos[1] < 480:
            # get distance from depth sensor to fingertip
            index_dis = depth_img[index_pos[1]][index_pos[0]]

            # control the mouse only when it is closer than max_distance
            if index_dis <= max_dis:
                # convert coordinates
                cur_index = get_interp_pos(index_pos, frame_reduc, cam_width, cam_height, scr_width, scr_height)
                # smoothen values
                # if cur point is initial point, do not smoothening
                if is_initial:
                    prev_index = cur_index

                else:
                    cur_index = get_smoothen_pos(prev_index, cur_index, smoothening)

                # calculate distance between current cursor position and last cursor updated position
                vir_dis = get_distance(cur_index, stopped_index)

                # -----------------------------------------------------------------------------------------
                # Index and Middle Finger is stretched: Draw Mode
                # -----------------------------------------------------------------------------------------
                # if index and middle finger are stretched, draw line in the palette
                if fingers[1] == 1 and fingers[2] == 1:
                    # to reduce effect of vibration
                    if vir_dis > vib_dis:
                        # move Mouse
                        # mouse.set_pos(0, 0): top right, (scr_width, scr_height): bottom left
                        cv2.circle(color_img, (index_pos[0], index_pos[1]), 5, (255, 0, 255), cv2.FILLED)
                        # //2 operation: palette is quarter of the screen size
                        if not is_initial:
                            paint.draw(is_initial, [prev_index[0] // 2, prev_index[1] // 2],
                                       [cur_index[0] // 2, cur_index[1] // 2], velocity)
                        else:
                            is_initial = False
                        stopped_index = prev_index

                    # calculate distance between current position and previous position
                    # this operation used in calculating velocity
                    distance = get_distance(cur_index, prev_index)
                    # update previous position to current position
                    prev_index = cur_index
                # -----------------------------------------------------------------------------------------
                # Index or Middle Finger is bend: Not Draw Mode
                # -----------------------------------------------------------------------------------------
                else:
                    # make initial flag True
                    is_initial = True

                # show distance between current cursor position and last cursor updated position in color image
                cv2.putText(color_img_reduc, str(vir_dis) + 'pixel', (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_img_reduc = cv2.applyColorMap(cv2.convertScaleAbs(depth_img_reduc, alpha=0.03), cv2.COLORMAP_JET)

    # calculate fps and velocity(pixel / sec / 50)
    cur_time = time.time()
    fps = 1 / (cur_time - past_time)
    velocity = int(distance / (cur_time - past_time) / 50)
    past_time = cur_time

    # show fps and velocity in color image
    cv2.putText(color_img_reduc, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(color_img_reduc, str(velocity) + 'pixel / second / scale_factor',
                (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # display color and depth image
    cv2.imshow("color_img_reduc", color_img_reduc)
    cv2.imshow("depth_img_reduc", depth_img_reduc)

    # ESC breaks the while loop
    pressed_key = cv2.waitKey(1)
    if pressed_key == 27:
        break

# release camera and close the window
cv2.destroyAllWindows()
pipeline.stop()
