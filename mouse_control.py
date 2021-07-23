# mouse_control.py

import win32api, win32con
from win32api import GetSystemMetrics

# The width of the screen of the primary display monitor, in pixels.
wScr = GetSystemMetrics(0)
# The height of the screen of the primary display monitor, in pixels. 
hScr = GetSystemMetrics(1)


class Mouse:
    SCROLL_INVERSE_GAIN = 20

    def __init__(self):
        self.clicked = False
        self.scrolling = False
        self.init_x = 0  # for scrolling
        self.init_y = 0
        self.x, self.y = win32api.GetCursorPos()

    # left mouse click
    def left_click(self):
        if self.clicked:
            return
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, self.x, self.y, 0, 0)  # click is true
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, self.x, self.y, 0, 0)

    # left mouse press
    def left_press(self):
        if self.clicked:
            return
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, self.x, self.y, 0, 0)
        self.init_x = self.x
        self.init_y = self.y
        self.clicked = True

    # left mouse unpress
    def left_unpress(self):
        if not self.clicked:
            return
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, self.x, self.y, 0, 0)
        self.clicked = False

    # right mouse click
    def right_click(self):
        if self.clicked:
            return
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, self.x, self.y, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, self.x, self.y, 0, 0)

    # right mouse press
    def right_press(self):
        if self.clicked:
            return
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, self.x, self.y, 0, 0)
        self.init_x = self.x
        self.init_y = self.y
        self.clicked = True

    # right mouse unpress
    def right_unpress(self):
        if not self.clicked:
            return
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, self.x, self.y, 0, 0)
        self.clicked = False

    # set cursor position
    def set_pos(self, x, y):
        # x, y must be integer
        x, y = int(x), int(y)
        if x < 0:
            x = 0;
        elif x >= wScr:
            x = wScr;
        if y < 0:
            y = 0;
        elif y >= hScr:
            y = hScr;
        point = (x, y)

        if not self.clicked:
            self.x, self.y = x, y

        # x, y must be integer
        win32api.SetCursorPos((self.x, self.y))

    # get cursor x, y position
    def get_pos(self):
        return win32api.GetCursorPos()

    # reset mouse condition
    def reset(self):
        self.left_unpress()
        self.right_unpress()
        self.__init__()