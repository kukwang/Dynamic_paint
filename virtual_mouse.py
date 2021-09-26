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
# helper function
# -----------------------------------------------------------------------------------------
# calculate distance between two points
def get_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (1 / 2)


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
prev_index_x, prev_index_y = 0, 0             # previous cursor position
cur_index_x, cur_index_y = 0, 0               # current cursor position
stopped_index_x, stopped_index_y = 0, 0       # last updated cursor position

vib_dis = 4                         # vibration distance threshold
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
#mouse = mouse_control.Mouse()
paint = painter_class.Paint(width=int(mouse_control.scr_width), height=int(mouse_control.scr_height))

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
    # chagne datatype (unit16 -> unit8)
    depth_img_u8 = depth_img.astype(np.uint8)

    # all filters have kernal size: 3x3
    # apply average burring
    depth_img_avg = cv2.blur(depth_img_u8, (3, 3))
    # apply gaussian blurring
    depth_img_gaussian = cv2.GaussianBlur(depth_img_u8, (3, 3), 0)
    # apply median blurring
    depth_img_median = cv2.medianBlur(depth_img_u8, 3)
    # apply bilateral blurring
    depth_img_bilateral = cv2.bilateralFilter(depth_img_u8, 3, 75, 75)
    # -----------------------------------------------------------------------------------------

    color_img_reduc = color_img[frame_reduc:cam_height - frame_reduc, frame_reduc:cam_width - frame_reduc]

    # find hand and its landmarks
    color_img = detector.find_hands(color_img, draw=True)
    lmList, _ = detector.find_position(color_img, draw=False)

    # if hand is detected, proceed to the next process
    if len(lmList) != 0:
        # get the fingertip of the index and middle fingers
        index_x, index_y = lmList[8][1:]
        middle_x, middle_y = lmList[12][1:]

        # check if fingers are up
        fingers = detector.fingers_up()
        # show fingertip sense area in the color image
        cv2.rectangle(color_img, (frame_reduc, frame_reduc),
                      (cam_width - frame_reduc, cam_height - frame_reduc), (255, 0, 255), 2)

        # to prevent indexError, limit the position of index fingertip point
        # sometimes index fingertip position
        if 0 <= index_x < 640 and 0 <= index_y < 480:
            # get distance from depth sensor to fingertip
            index_dis = depth_img_u8[index_y][index_x]

            # control the mouse only when it is closer than max_distance
            if index_dis <= max_dis:
                # convert coordinates
                cur_index_x = int(np.interp(index_x, (frame_reduc, cam_width - frame_reduc), (0, mouse_control.scr_width)))
                cur_index_y = int(np.interp(index_y, (frame_reduc, cam_height - frame_reduc), (0, mouse_control.scr_height)))
                # smoothen values
                # if cur point is initial point, do not smoothening
                if is_initial:
                    prev_index_x, prev_index_y = cur_index_x, cur_index_y

                else:
                    cur_index_x = int(prev_index_x + (cur_index_x - prev_index_x) / smoothening)
                    cur_index_y = int(prev_index_y + (cur_index_y - prev_index_y) / smoothening)

                # calculate distance between current cursor position and last cursor updated position
                vir_dis = get_distance([cur_index_x, cur_index_y], [stopped_index_x, stopped_index_y])

                # -----------------------------------------------------------------------------------------
                # Index and Middle Finger is stretched: Draw Mode
                # -----------------------------------------------------------------------------------------
                # if middle finger is stretched, draw line in the palette
                if fingers[1] == 1 and fingers[2] == 1:
                    # to reduce effect of vibration
                    if vir_dis > vib_dis:
                        # move Mouse
                        # mouse.set_pos(0, 0): top right, (scr_width, scr_height): bottom left
                        #mouse.set_pos(cur_index_x, cur_index_y)
                        cv2.circle(color_img, (index_x, index_y), 5, (255, 0, 255), cv2.FILLED)
                        # //2 operation: palette is quarter of the screen size
                        if not is_initial:
                            paint.draw(is_initial, [prev_index_x // 2, prev_index_y // 2],
                                       [cur_index_x // 2, cur_index_y // 2], velocity)
                        else:
                            is_initial = False
                        stopped_index_x, stopped_index_y = prev_index_x, prev_index_y

                    # calculate distance between current position and previous position
                    # this operation used in calculating velocity
                    distance = get_distance([cur_index_x, cur_index_y], [prev_index_x, prev_index_y])
                    # update previous position to current position
                    prev_index_x, prev_index_y = cur_index_x, cur_index_y

                # -----------------------------------------------------------------------------------------
                # Index or Middle Finger is bend: Not Draw Mode
                # -----------------------------------------------------------------------------------------
                else:
                    # make initial flag True
                    is_initial = True

                # show distance between current cursor position and last cursor updated position in color image
                cv2.putText(color_img_reduc, str(vir_dis) + 'pixel', (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

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
    #cv2.imshow("color_img", color_img)
    cv2.imshow("color_img_reduc", color_img_reduc)
    cv2.imshow("depth_img", depth_img)

    merged_blur_result = np.hstack((depth_img_avg, depth_img_gaussian))
    merged_blur_result2 = np.hstack((depth_img_median, depth_img_bilateral))

    # ESC breaks the while loop
    pressed_key = cv2.waitKey(1)
    if pressed_key == 27:
        break

# release camera and close the window
cv2.destroyAllWindows()
pipeline.stop()
