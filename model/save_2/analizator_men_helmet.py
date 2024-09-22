def is_helmet_on_head(helmet, person):
    helmet_x, helmet_y = helmet['center']
    helmet_width, helmet_height = helmet['size']

    person_x, person_y = person['center']
    person_width, person_height = person['size']

    if (helmet_x > person_x - person_width / 2 and
            helmet_x < person_x + person_width / 2 and
            helmet_y > person_y - person_height * 5 / 10 and
            helmet_y < person_y - person_height * 2 / 10):
        return True
    return False

# Преобразование бокса YOLO в центр, ширину и высоту
def box_to_center_size(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    return {'center': (center_x, center_y), 'size': (width, height)}

# Основной метод обработки предсказаний
def analyze_predictions(results):
    helmets = []
    people = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Получаем боксы
        classes = result.boxes.cls.cpu().numpy()  # Классы объектов
        for i, box in enumerate(boxes):
            cls = classes[i]
            if cls == 0:  # Класс 0: человек
                people.append(box_to_center_size(box))
            elif cls == 1:  # Класс 1: шлем
                helmets.append(box_to_center_size(box))

    # Анализируем, находится ли шлем на голове каждого человека
    for person in people:
        helmet_found = False
        for helmet in helmets:
            if is_helmet_on_head(helmet, person):
                print("Шлем на голове человека.")
                helmet_found = True
                break
        if not helmet_found:
            print("Шлем НЕ на голове человека.")

# Пример вызова анализа с результатами модели
# Допустим, `results` — это результат предсказания модели YOLOv8
# results = model.predict(image)
# analyze_predictions(results)