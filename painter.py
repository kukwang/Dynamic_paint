# 일단 만들어본거
import cv2
import numpy as np
from scipy import interpolate

# 콜백 함수에서 사용되는 매개변수: event, x, y, flags, param
# event: 윈도우에서 발생하는 이벤트를 의미
# x, y: 마우스의 좌표
# flags: event와 함께 활용되는 역할로 특수한 상태를 확인하는 용도
# param: 마우스 콜백 설정 함수에서 함께 전달되는 사용자 정의 데이터를 의미
def mouse_draw(event, x, y, flags, param):
    global radius, red, green, blue
    
    red = cv2.getTrackbarPos("Red", "palette")
    blue = cv2.getTrackbarPos("Blue", "palette")
    green = cv2.getTrackbarPos("Green", "palette")

    # 마우스 왼쪽 버튼이 눌러져 있을 때 검은 원을 그림
    point = []
    if flags == cv2.EVENT_FLAG_LBUTTON:
        # point = [x, y]
        cv2.circle(param, (x, y), radius, (blue, green, red), cv2.FILLED)
        cv2.imshow("palette", src)

    # 만약, event가 마우스 스크롤을 조작했다면, 다시 하위 분기문(if)을 생성하여 나눔
    # event가 마우스 스크롤 이벤트일 때, flag는 마우스 스크롤의 방향을 나타냄
    # flag가 양수라면 스크롤 업, 음수라면 스크롤 다운
    # 마우스 스크롤 업 이벤트일 때는 반지름(radius)를 증가시키고, 낮을 때에는 반지름을 감소
    # 단, 반지름이 1보다 작지 않게 설정하기 위해, radius > 1 조건으로 검사        
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            radius += 1
        elif radius > 1:
            radius -= 1

def create_smooth_curve(points):
    x = points[0, :]
    y = points[1, :]
    delta = 4

    t = np.arange(0, len(x), 1)
    spl_x = interpolate.InterpolatedUnivariateSpline(t, x)
    spl_y = interpolate.InterpolatedUnivariateSpline(t, y)
    t_inp = np.arange(0, len(x), 1/delta)
    x_inp = np.trunc(spl_x(t_inp)).astype(np.uint32)
    y_inp = np.trunc(spl_y(t_inp)).astype(np.uint32)
    curve_points = (x_inp, y_inp)

    return curve_points

# radius: 붓 반지름
radius = 1
red, green, blue = 0, 0, 0

src = np.full((960, 1024, 3), 255, dtype=np.uint8)

cv2.namedWindow("palette")
cv2.createTrackbar("Red", "palette", 0, 255, lambda x : x)
cv2.createTrackbar("Green", "palette", 0, 255, lambda x : x)
cv2.createTrackbar("Blue", "palette", 0, 255, lambda x : x)

cv2.setTrackbarPos("Red", "palette", 0)
cv2.setTrackbarPos("Green", "palette", 0)
cv2.setTrackbarPos("Blue", "palette", 0)

cv2.imshow("palette", src)

# cv2.setMouseCallback(윈도우, 콜백 함수, 사용자 정의 데이터): 마우스 콜백 설정
# 윈도우: 미리 생성되어 있는 윈도우의 이름
# 콜백 함수: 마우스 이벤트가 발생했을 때, 전달할 함수를 의미
# 사용자 정의 데이터: 마우스 이벤트로 전달할 때, 함께 전달할 사용자 정의 데이터 의미
cv2.setMouseCallback("palette", mouse_draw, src)

cv2.waitKey()
cv2.destroyAllWindows()