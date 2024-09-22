import time
from threading import Thread

import cv2
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from model.analizator_model import Frame, Analyser, FrameBuffer, RedLine
from model.sign_detector import SignDetector
from ui.window_interface import Ui_MainWindow



class MainWindow(QMainWindow, Ui_MainWindow):
    fps: int = None
    video_capture: cv2.VideoCapture = None
    video_paused = True
    file_name: str
    playing_video_thread = None
    source_image: np.ndarray = None
    resized = QtCore.pyqtSignal()
    sign_detector = SignDetector(r'yolov8n_custom\weights\best.pt')

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.load_file_action.triggered.connect(self.load_file)
        self.start_btn.clicked.connect(self.start_btn_click)
        self.start_pause_action.triggered.connect(self.start_btn_click)
        self.camera_action.triggered.connect(self.switch_camera_mode)
        self.enable_signs_detection_checkbox.setChecked(True)
        self.quit_action.triggered.connect(self.close)
        self.resized.connect(self.update_image_view)
        self.video_paused = True

        self.frame_buffer = FrameBuffer(size=20)  # Создаем буфер для последних 10 кадров
        self.k = None
        self.b = None

    def resizeEvent(self, event):
        self.resized.emit()
        event.accept()
        return super(MainWindow, self).resizeEvent(event)

    def closeEvent(self, event):
        self.video_paused = True
        event.accept()

    def update_image_view(self):
        if self.source_image is not None:
            self.update_image(self.source_image)

    def switch_camera_mode(self):
        self.video_capture = cv2.VideoCapture(0)
        self.fps = 30
        self.start_btn.setEnabled(True)

    def load_file(self):
        options = QFileDialog.Options()
        file_filter: str
        file_name, file_filter = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Video files (*.mp4 *.avi *.mkv);;All files (*)", options=options)

        if not file_name:
            return

        self.start_btn.setEnabled(True)

        # self.source_image = cv2.imread(file_name)
        # self.update_image(self.source_image)

        if file_filter == "Video files (*.mp4 *.avi *.mkv)":
            self.file_name = file_name
            self.video_capture = cv2.VideoCapture(file_name)
            self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

    def start_btn_click(self):
        self.video_paused = not self.video_paused

        if self.video_paused:
            self.start_btn.setText('Пуск')
        else:
            self.start_btn.setText('Пауза')
            self.playing_video_thread = Thread(target=self.play_video)
            self.playing_video_thread.start()

    def play_video(self):
        delta_time = time.time()
        number_frame = 0  # Номер кадра 1,2,3,4 ... 200,201...
        frames_counter = 0
        while not self.video_paused:
            # for _ in range(0, 29):
            _, frame = self.video_capture.read()
            if frame is None:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            number_frame += 1

            # self.source_image = frame
            start_time = time.time()
            self.process_frame(frame, number_frame)
            end_time = time.time()
            if end_time - delta_time >= 1:
                print(f"FPS: {frames_counter} number_frame:{number_frame}")
                delta_time = end_time
                frames_counter = 0

            frames_counter += 1

            time.sleep(max(0.0, 1 / self.fps - end_time + start_time))

        # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_frame(self, image, number_frame):
        redline = RedLine()
        if self.k is None:
            self.k, self.b = redline.find_red_line_with_angle(image)
        #     man_coords = (207, 564)  # xmax ymax
        #
        #     print(is_man_in_room(k, b, man_coords))

        if self.enable_signs_detection_checkbox.isChecked():
            # Получаем результат предсказания от модели
            res = self.sign_detector.predict(image)

            # Извлекаем боксы, классы и уверенности
            boxes = res.boxes.xyxy  # Координаты боксов в формате [x_min, y_min, x_max, y_max]
            confidences = res.boxes.conf  # Уверенности предсказаний
            classes = res.boxes.cls  # Классы объектов

            # Преобразуем данные в нужный формат для создания объекта Frame
            boxes_list = boxes.tolist()  # Преобразуем тензор в список для удобства
            confidences_list = confidences.tolist()  # Преобразуем тензор уверенности в список
            classes_list = classes.tolist()  # Преобразуем тензор классов в список

            # Создаем объект Frame для классификации объектов
            frame = Frame(boxes_list, confidences_list, classes_list)

            # Сохраняем кадр в буфер
            if number_frame % 5 == 0:
                self.frame_buffer.add_frame(frame)

            # Выводим результаты классификации в консоль
            if number_frame % 30 == 0:
                frame.print_results()
                analyser = Analyser(self.frame_buffer, frame, self.k, self.b)
                analyser.analyse_door()
                analyser.analyse_helmets()  # Добавляем вызов анализа касок
                analyser.analyse_robot_movement()

            # Обновляем изображение с предсказаниями
            image = res.plot()  # Визуализация предсказаний на изображении
        self.update_image(image)

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_to_qt(cv_img)
        scaled_image = qt_img.scaled(self.image_view.width(), self.image_view.height(), Qt.KeepAspectRatio)
        self.image_view.setPixmap(QPixmap.fromImage(scaled_image))

    @staticmethod
    def convert_cv_to_qt(cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return converted_image
