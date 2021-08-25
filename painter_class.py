# painter_class.py
import cv2
import numpy as np


class Paint:
    # paint area is quarter of the screen size
    def __init__(self, width=1920, height=1080, radius_smoothening = 7):
        self.bgr = [0, 0, 0]                                                # rgb color of brush
        self.is_initial = True                                              # check if this class run first time
        self.max_radius = 30                                                # max radius of brush
        self.src = np.full((height//2, width//2, 3), 255, dtype=np.uint8)   # paint in this area
        self.radius_smoothening = radius_smoothening
        self.prev_radius, self.cur_radius = 0, 0
        cv2.imshow("palette", self.src)

    def draw(self, clicked, prev_loc, cur_loc, velocity, is_steady):
        radius = self.change_radius(velocity, is_steady)

        # 마우스 왼쪽 버튼이 눌러져 있을 때 검은 원을 그림
        if clicked:
            if self.is_initial:
                self.is_initial = False
            else:
                cv2.line(self.src, prev_loc, cur_loc, self.bgr, thickness=radius, lineType=cv2.LINE_AA)

            cv2.imshow("palette", self.src)

        # 만약, event가 마우스 스크롤을 조작했다면, 다시 하위 분기문(if)을 생성하여 나눔
        # event가 마우스 스크롤 이벤트일 때, flag는 마우스 스크롤의 방향을 나타냄
        # flag가 양수라면 스크롤 업, 음수라면 스크롤 다운
        # 마우스 스크롤 업 이벤트일 때는 반지름(radius)를 증가시키고, 낮을 때에는 반지름을 감소
        # 단, 반지름이 1보다 작지 않게 설정하기 위해, radius > 1 조건으로 검사

        else:
            self.is_initial = True

        print('v : ' + str(velocity) + ' r : ' + str(radius))

    def change_radius(self, velocity, is_steady):
        scaling = 3
        sensitivity = 0.1
        velo_shift = 30

        if is_steady:
            return
        else:
            # update current radius using current velocity
            self.cur_radius = int(scaling * np.exp((velo_shift - velocity) * sensitivity))
            # to make smooth
            self.cur_radius = int(self.prev_radius + (self.cur_radius - self.prev_radius) / self.radius_smoothening)
            self.prev_radius = self.cur_radius
            #self.cur_radius = int(velo_shift - velocity)
            if self.cur_radius >= self.max_radius:
                return self.max_radius
            elif self.cur_radius < 1:
                return 1
            else:
                return self.cur_radius
