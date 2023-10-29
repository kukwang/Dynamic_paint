# Dynamic Paint

## Introduction
Virtual mouse using Intel RealSense Depth Camera D435. Use [mediapipe](https://github.com/google/mediapipe)

## Process
1. Get RGB and Depth information from depth camera
2. Detect hand and its position using mediapipe framework
3. Determines whether the hand is within a defined area using depth information 
4. Draw circle according to speed of the finger
5. Repeat 1-4 until stop

## Example
![example1](https://github.com/kukwang/dynamic_paint/assets/52880303/aded3e94-35db-4cf6-9d74-13230d06c991)

![example2](https://github.com/kukwang/dynamic_paint/assets/52880303/1373cc94-a7e5-4778-ac8d-1022b)
