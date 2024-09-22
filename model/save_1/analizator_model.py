import numpy as np


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
        print(f"Количество классов {len(classes)}")
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


class Analyser:

    def __init__(self, frame: Frame):
        self.frame = frame

    def analyse_door(self):
        if len(self.frame.door_close) > 0:
            print("STATUS DOOR CLOSED")
        elif len(self.frame.door_open) > 0:
            print("STATUS DOOR OPENED")
        else:
            print("STATUS DOOR CLOSED (NOT FOUND DOOR)")
