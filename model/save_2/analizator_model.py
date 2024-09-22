import numpy as np
from collections import deque


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
            boxplot = Boxplot(x_min, y_min, width, height, confidences[i])

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
    def __init__(self, x=None, y=None, width=None, height=None, confidences=None):
        self.x = x
        self.y = y
        self.width = width
        self.high = height
        # Центр бокса
        self.x_center = x + width / 2
        self.y_center = y + height / 2
        self.confidences = confidences


class FrameBuffer:
    def __init__(self, size=10):
        self.buffer = deque(maxlen=size)  # Создаем очередь с максимальным размером 10

    def add_frame(self, frame):
        self.buffer.append(frame)  # Добавляем новый кадр в очередь

    def get_frames(self):
        return list(self.buffer)  # Возвращаем список последних сохраненных кадров


class Analyser:
    def __init__(self, frame_buffer, frame: Frame):
        self.frame_buffer = frame_buffer  # Теперь используем буфер кадров вместо одного кадра
        self.frame = frame

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

