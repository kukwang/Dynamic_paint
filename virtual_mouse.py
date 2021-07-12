# AiVirtualMouseProject.py
# using win32api

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
max_distance = 0.5        # 0.5m

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

# assign handDetector and Mouse class
detector = hand_tracing.handDetector(maxHands=1)
mouse = mouse_control.Mouse()

# create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames
align_to = rs.stream.color
align = rs.align(align_to)

while True:
#i = 0
#while i == 0:
#    i = 1
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
    color_img = detector.findHands(color_img, draw=True)
    lmList, bbox = detector.findPosition(color_img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(color_img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if x1 >= 0 and y1 >= 0 and x1 < 480 and y1 < 640:
            # get distance from depth sensor to fingertip, not stable
            distance1 = aligned_depth_frame.get_distance(x1, y1)
            # 4. Only Index Finger : Moving Mode
            # index finger is stretched and distance <= max distance, middle finger is bent
            if fingers[1] == 1 and fingers[2] == 0 and distance1 <= max_distance:
                # 5. Convert Coordinates
                clocX = int(np.interp(x1, (frameR, wCam - frameR), (0, wScr)))
                clocY = int(np.interp(y1, (frameR, hCam - frameR), (0, hScr)))

                # 6. Move Mouse
                # mouse.set_pos(0, 0): top right
                # mouse.set_pos(wScr, hScr): bottom left
                mouse.set_pos(clocX, clocY)
                cv2.circle(color_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(depth_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # 7. Both Index and middle fingers are up : Clicking Mode
            # index finger and middle finger are stretched and distance <= max distance
            if fingers[1] == 1 and fingers[2] == 1:

                # 8. Find distance between fingers
                length, color_img, lineInfo = detector.findDistance(8, 12, color_img)
                _, depth_img, _ = detector.findDistance(8, 12, depth_img)

                if x2 >= 0 and y2 >= 0 and x2 < 480 and y2 < 640:
                    # get distance from depth sensor to fingertip, not stable
                    distance2 = aligned_depth_frame.get_distance(x2, y2)

                    # 9. Click mouse if distance short
                    if length < 40 and distance1 <= max_distance and distance2 <= max_distance:
                        cv2.circle(color_img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        cv2.circle(depth_img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        mouse.left_click()

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # dimension: 640x480 -> 640x480x3
    depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.1), cv2.COLORMAP_JET)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    # 10. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(color_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(depth_img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 11. Display
    cv2.imshow("color_img", color_img)
    cv2.imshow("depth_img", depth_img)

    # ESC break the while loop
    if cv2.waitKey(1) == 27:
        break

# release camera and close the window
pipeline.stop()
cv2.destroyAllWindows()
