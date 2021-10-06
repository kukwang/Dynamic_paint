# painter_class.py
import cv2
import numpy as np


class Paint:
    # palette is quarter of the screen size
    def __init__(self, width=1920, height=1080):
        self.bgr = [0, 0, 0]    # rgb color of brush
        # palette: size is (height//2 * width//2), 3 channels, color is white
        self.src = np.full((height//2, width//2, 3), 255, dtype=np.uint8)
        self.prev_radius, self.cur_radius = 0, 0    # previous and current radius, initial value is 0

        # if this class called, show palette
        cv2.imshow("palette", self.src)

    # function that draw line in the palette(self.src)
    def draw(self, is_initial, prev_loc=0, cur_loc=0, velocity=0):
        # calculate radius
        radius = self.change_radius(velocity)

        # if cur mouse position is not first position, draw line between prev and cur position at palette
        if not is_initial:
            cv2.line(self.src, prev_loc, cur_loc, self.bgr, thickness=radius, lineType=cv2.LINE_AA)
        print("velocity:", velocity, "radius:", radius)
        cv2.imshow("palette", self.src)

    # function that change brush size according to velocity of index fingertip
    def change_radius(self, velocity):
        # parameters that we use to calculate brush size(radius)
        scaling = 3
        sensitivity = 0.09
        velo_shift = 30
        radius_smoothening = 7

        max_radius = 50     # max radius of brush
        min_radius = 3      # min radius of brush

        # update current radius using current velocity
        self.cur_radius = int(scaling * np.exp((velo_shift - velocity) * sensitivity))
        # to make change of the radius smooth,
        # reflect only some of the changes between previous and current radius
        self.cur_radius = int(self.prev_radius + (self.cur_radius - self.prev_radius) / radius_smoothening)
        # Update previous radius to current radius
        self.prev_radius = self.cur_radius

        # if current radius is larger than max_radius we set, radius is max_radius
        # if current radius is smaller than min_radius we set, radius is min_radius
        # else, radius is current radius
        if self.cur_radius >= max_radius:
            return max_radius
        elif self.cur_radius < min_radius:
            return min_radius
        else:
            return self.cur_radius
