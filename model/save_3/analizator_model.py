import numpy as np
from collections import deque
import cv2
import numpy as np


class RedLine:
    def __init__(self):
        self.target_angle = -22
        self.angle_tolerance = 5

    @staticmethod
    def is_man_in_room(k, b, man_coords: tuple):  # man_coords xmax, ymax
        return man_coords[1] >= k * man_coords[0] + b

    def find_red_line_with_angle(self, image):
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

                if abs(angle - self.target_angle) <= self.angle_tolerance:
                    k = (y2 - y1) / (x2 - x1)
                    return k, y1 - k * x1


# if __name__ == '__main__':
#     image_path = '123.png'
#     image = cv2.imread(image_path)
#     target_angle = -22
#     angle_tolerance = 5
#
#     k, b = find_red_line_with_angle(image, target_angle, angle_tolerance)
#     man_coords = (207, 564)  # xmax ymax
#
#     print(is_man_in_room(k, b, man_coords))


class Frame:
    def __init__(self, boxes, confidences, classes):
        # Инициализация значений для каждого класса
        self.door_close = []
        self.door_open = []
        self.helmet = []
        self.men = []
        self.robot = []

        # классы
        # 0 = door_close
        # 1 = door_open
        # 2 = helmet
        # 3 = men
        # 4 = robot
        # print(f"Количество классов {len(classes)}")
        for i, cls in enumerate(classes):
            # Разбираем координаты бокса
            x_min, y_min, x_max, y_max = boxes[i]
            width = x_max - x_min
            height = y_max - y_min
            boxplot = Boxplot(x_min, y_min, x_max, y_max, width, height, confidences[i])

            # Распределение по классам
            if cls == 0:
                self.door_close.append(boxplot)
            elif cls == 1:
                self.door_open.append(boxplot)
            elif cls == 2:
                self.helmet.append(boxplot)
            elif cls == 3:
                self.men.append(boxplot)
            elif cls == 4:
                self.robot.append(boxplot)

    def print_results(self):
        """
        Выводит классифицированные результаты в консоль.
        """
        print("\n\n\nСтатистика кадра")
        print(f"Closed doors: {len(self.door_close)}")
        print(f"Open doors: {len(self.door_open)}")
        print(f"Helmets: {len(self.helmet)}")
        print(f"Men: {len(self.men)}")
        print(f"Robots: {len(self.robot)}")


class Boxplot:
    def __init__(self, x_min=None, y_min=None, x_max=None, y_max=None, width=None, height=None, confidences=None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.width = width
        self.high = height
        # Центр бокса
        self.x_center = x_min + width / 2
        self.y_center = y_min + height / 2
        self.confidences = confidences


class FrameBuffer:
    def __init__(self, size=10):
        self.buffer = deque(maxlen=size)  # Создаем очередь с максимальным размером 10

    def add_frame(self, frame):
        self.buffer.append(frame)  # Добавляем новый кадр в очередь

    def get_frames(self):
        return list(self.buffer)  # Возвращаем список последних сохраненных кадров


class Analyser:
    def __init__(self, frame_buffer, frame: Frame, k_angel, b_number):
        self.frame_buffer = frame_buffer  # Теперь используем буфер кадров вместо одного кадра
        self.frame = frame
        self.k_angel = k_angel
        self.b_number = b_number

    def analyse_door(self):
        last_frames = self.frame_buffer.get_frames()[-5:]  # Получаем последние 5 кадров

        # Флаги для проверки состояния дверей
        door_closed_found = False
        door_open_all = True

        # Проверяем состояние на последних 5 кадрах
        for frame in last_frames:
            if len(frame.door_close) > 0:
                door_closed_found = True  # Найдена закрытая дверь
            if len(frame.door_open) == 0:
                door_open_all = False  # На этом кадре дверь не открыта

        # Логика определения статуса двери
        if door_closed_found:
            print("STATUS DOOR CLOSED")
        elif door_open_all:
            print("STATUS DOOR OPENED")
        else:
            print("STATUS DOOR CLOSED (NOT FOUND DOOR)")

    def analyse_helmets(self):
        # Проходим по всем людям
        for person in self.frame.men:
            person_x, person_y = person.x_center, person.y_center
            person_max_x, person_max_y = person.x_max, person.y_max
            person_width, person_height = person.width, person.high

            helmet_found = False  # Флаг для проверки, есть ли у человека каска

            # Проходим по всем каскам
            for helmet in self.frame.helmet:
                helmet_x, helmet_y = helmet.x_center, helmet.y_center

                # Проверяем, находится ли каска в области головы человека
                if (helmet_x > person_x - person_width / 2 and
                        helmet_x < person_x + person_width / 2 and
                        helmet_y > person_y - person_height * 5 / 10 and
                        helmet_y < person_y - person_height * 2 / 10):
                    print("Человек в каске")
                    helmet_found = True
                    break  # Каска найдена, дальнейшие проверки не нужны

            # Если каска не найдена, выводим сообщение, что человек без каски
            if not helmet_found:
                print("Человек без каски")
                if RedLine().is_man_in_room(self.k_angel, self.b_number, (person_max_x, person_max_y)):
                    print("Человек в опасной зоне")
