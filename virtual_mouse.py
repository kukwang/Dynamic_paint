"""
virtual_mouse.py
By: Murtaza Hassan
Youtube: https://www.youtube.com/watch?v=8gPONnGIPgw
Website: https://www.computervision.zone/courses/ai-virtual-mouse/
Modified by kukwang
Using win32api
"""

import cv2
import numpy as np
import time
import hand_tracing
import mouse_control

# -----------------------------------------------------------------------------------------
# parameters
# -----------------------------------------------------------------------------------------
wCam, hCam = 640, 480   # window size
frameR = 100            # Frame Reduction
smoothening = 7
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
distance = 0
x1, y1 = 0, 0

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

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
max_distance_in_meters = 2 #1 meter
max_distance = max_distance_in_meters / depth_scale
threshold_dis1 = 0.75/depth_scale
threshold_dis2 = 1.25/depth_scale

# create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames
align_to = rs.stream.color
align = rs.align(align_to)

# assign HandDetector and Mouse class
detector = hand_tracing.HandDetector(maxHands=1)
mouse = mouse_control.Mouse()

cnt = 0
while True:
    cnt += 1
    # -----------------------------------------------------------------------------------------
    # get color, depth image from depth camera
    # -----------------------------------------------------------------------------------------
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    # frames.get_depth_frame() is a 640x360 depth image
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(aligned_depth_frame.get_data())
    color_img = cv2.flip(color_img, 1)
    depth_img = cv2.flip(depth_img, 1)
    # -----------------------------------------------------------------------------------------

    # 1. Find hand Landmarks
    color_img = detector.find_hands(color_img, draw=True)
    lmList, bbox = detector.find_position(color_img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(color_img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        
        # prevent indexError
        if x1 >= 0 and y1 >= 0 and x1 < 640 and y1 < 480:
            # get distance from depth sensor to fingertip
            distance1 = depth_img[y1][x1]

            # control the mouse only when it is closer than max_distance
            if distance1 <= max_distance:
                # 4. Only Index Finger : Moving Mode
                # index finger is stretched and distance <= max distance, middle finger is bent
                if fingers[1] == 1 and fingers[2] == 0:
                    # 5. Convert Coordinates
                    clocX = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    clocY = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                    # 6. Smoothen Values
                    clocX = plocX + (clocX - plocX) / smoothening
                    clocY = plocY + (clocY - plocY) / smoothening

                    # to remove effect of vibration
                    if velocity > 0:
                        # 7. Move Mouse
                        # mouse.set_pos(0, 0): top right
                        # mouse.set_pos(wScr, hScr): bottom left
                        mouse.set_pos(clocX, clocY)
                        cv2.circle(color_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        cv2.circle(depth_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        
                    # --------------------------------------------------------------
                    # calculate distance
                    # --------------------------------------------------------------
                    distance = mouse_control.cal_distance([clocX, clocY], [plocX, plocY]) 
                    
                    plocX, plocY = clocX, clocY

                # prevent indexError
                if x2 >= 0 and y2 >= 0 and x2 < 640 and y2 < 480:
                    # 8. Both Index and middle fingers are up : Clicking Mode
                    # index finger and middle finger are stretched and distance <= max distance
                    if fingers[1] == 1 and fingers[2] == 1:

                        # 9. Find distance between fingers
                        length, color_img, lineInfo = detector.find_distance(8, 12, color_img)
                        _, depth_img, _ = detector.find_distance(8, 12, depth_img)

                        # 10. Click mouse if distance short
                        if length < 40:
                            cv2.circle(color_img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            cv2.circle(depth_img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            mouse.left_click()

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # dimension: 640x480 -> 640x480x3
    depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    # 11. fps and velocity(pixel / sec / 50)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    velocity = int(distance / (cTime - pTime) / 50)
    pTime = cTime
    cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(depth_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # -----------------------------------------------------------------------------------------
    # show velocity of the mouse
    # -----------------------------------------------------------------------------------------
    cv2.putText(color_img, str(velocity) + 'pixel / second / 50', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # 12. Display
    cv2.imshow("color_img", color_img)
    cv2.imshow("depth_img", depth_img)

    
    # ESC break the while loop
    if cv2.waitKey(1) == 27:
        break

# release camera and close the window
cv2.destroyAllWindows()
pipeline.stop()
