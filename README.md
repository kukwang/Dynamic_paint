# Dynamic Paint

## Introduction
Virtual mouse using Intel RealSense Depth Camera D435. We use [mediapipe](https://github.com/google/mediapipe) framework to detect hand.

## Process
1. Get RGB and Depth information from depth camera
2. Detect hand and its position using mediapipe framework
3. Determines whether the hand is within a defined area using depth information 
4. Draw circle according to speed of the finger
5. Repeat 1-4 until stop

## Examples
* Example 1

![Example1](https://github.com/kukwang/Dynamic_paint/assets/52880303/d050d94b-1bb3-475f-bc98-64dba48259ac)

* Example 2

![Example2](https://github.com/kukwang/Dynamic_paint/assets/52880303/8492f748-d8e4-4663-802c-7df7cf0c7238)
