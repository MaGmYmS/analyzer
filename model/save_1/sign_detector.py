import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from frame import Frame


class SignDetector:
    model: YOLO = None
    last_frames: np.ndarray = np.empty(8, dtype=Frame)

    def __init__(self, file_name):
        if file_name is None:
            raise 'wtf model file not found'
        self.model = YOLO(file_name)

    def predict(self, image: np.ndarray):
        res = self.model.predict(image, verbose=False)

        # boxes: Boxes = res[0].boxes.cpu().numpy()
        # centers = np.mean(boxes.xyxy.reshape((-1, 2, 2)), 1).astype(int)
        # self.last_frames[:-1] = self.last_frames[1:]
        # self.last_frames[-1] = Frame(boxes.cls[0], boxes.conf, centers[0])
        #
        # if self.last_frames[-2] is not None:
        #     coords_sum = np.zeros(2)
        #     multiplier = 1
        #
        #     for frame in self.last_frames:
        #         if frame is not None:
        #             coords_sum += frame.coords * multiplier
        #             multiplier *= -1

            # print(f'Разница координат 8 последних кадров: {sum(abs(coords_sum))}')
            # print(f'Вероятность 2 последних: {self.last_frames[-2].confidence}, {self.last_frames[-1].confidence}')
            # print(self.last_centers)
        return res[0]
