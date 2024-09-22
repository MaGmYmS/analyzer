import cv2
import numpy as np


def is_man_in_room(k, b, man_coords: tuple):
    return man_coords[1] >= k * man_coords[0] + b


def find_red_line_with_angle(image, target_angle, angle_tolerance=5):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)
    red_result = cv2.bitwise_and(image, image, mask=red_mask)

    gray = cv2.cvtColor(red_result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if angle < -90:
                angle += 180
            elif angle > 90:
                angle -= 180

            if abs(angle - target_angle) <= angle_tolerance:
                k = (y2 - y1) / (x2 - x1)
                return k, y1 - k * x1


if __name__ == '__main__':
    image_path = '123.png'
    image = cv2.imread(image_path)
    target_angle = -22
    angle_tolerance = 5

    k, b = find_red_line_with_angle(image, target_angle, angle_tolerance)
    man_coords = (207, 564)

    print(is_man_in_room(k, b, man_coords))
