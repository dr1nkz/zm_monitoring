import os
import cv2
import time
import argparse
import mysql.connector
from datetime import datetime

import numpy as np

from detector import YOLOv8
from utils import compute_polygon_intersection

# import zmes_hook_helpers.common_params as g


def parseurl(temp):
    if '://' in temp[-1]:
        temp = temp[-1]
    else:
        temp = temp[0] + '://' + temp[1] + ':' + temp[2] + temp[3]
    return temp


def parse_monitor_parameters(monitor_id):
    mydb = mysql.connector.connect(
        host='localhost',
        user='zmuser',
        password='zmpass',
        database='zm'
    )

    # ------Address------
    mycursor = mydb.cursor()

    mycursor.execute(
        f"SELECT Protocol, Host, Port, Path FROM Monitors WHERE ID={monitor_id}")

    address = mycursor.fetchall()[0]
    address = parseurl(address)
    # print(address, '\n')

    # ------Zones------
    mycursor.execute(f"SELECT Coords FROM Zones WHERE MonitorId={monitor_id}")

    zones = mycursor.fetchall()
    # print(zones)

    return address, zones
    # return 'http://192.168.0.39:8080/video', [('17,475 0,0 407,12 407,466',)]


def gen(monitor_id, duration=30):  # Длительность нахождения
    """
    Запуск модели
    """
    # Засекаем начало работы скрипта
    start_time = time.time()

    address, zones = parse_monitor_parameters(monitor_id)
    print(address)

    model_path = r'/var/lib/zmeventnotification/bin/my_detection/hf.onnx'
    yolov8_detector = YOLOv8(path=model_path,
                             conf_thres=0.3,
                             iou_thres=0.5)

    # cv2.namedWindow('stream', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(address)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    filename = '/var/lib/zmeventnotification/bin/my_detection/videos/monitor_' + \
        monitor_id + datetime.now().strftime(r'_%d.%m.%Y_%H:%M:%S') + '.mp4'
    out = cv2.VideoWriter(filename, fourcc, 30, (width, height))

    # Координаты территории
    parking_areas = []
    for zone in zones:
        bbox = [list(map(int, elem.split(','))) for elem in zone[0].split(' ')]
        parking_areas.append(bbox)
    print(parking_areas)

    # Сколько кадров подряд с пустым местом мы уже видели
    free_frames = 0

    while cap.isOpened():
        # Кадр с камеры
        ret, frame = cap.read()
        if not ret:
            break

        # Детектирование
        yolov8_detector(frame)
        # detected_img = frame
        detected_img = yolov8_detector.draw_detections(frame)
        bounding_boxes = np.array(yolov8_detector.get_boxes())
        if detected_img is None:
            continue

        # чтобы не ломалось iou
        if len(bounding_boxes) == 1 or bounding_boxes.shape[0] == 1:
            # bounding_boxes = np.array([bounding_boxes])
            bounding_boxes = np.array(bounding_boxes).reshape(1, -1)

        free_frames_temp = 0
        if bounding_boxes.shape[0] != 0:
            for bounding_box in bounding_boxes:
                intersections = compute_polygon_intersection(
                    bounding_box, parking_areas)
                max_intersection = np.max(intersections)
                # Для отладки параметра IoU
                # print(parked_drums_boxes[i])
                # print(max_IoU)
                if max_intersection > 0.35:
                    free_frames = 0
                    continue
                else:
                    free_frames_temp += 1
        else:
            free_frames += 1

        if free_frames_temp == bounding_boxes.shape[0]:
            free_frames += 1

        if free_frames > 100:
            break

        # cv2.imshow('stream', detected_img)
        out.write(detected_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cv2.destroyAllWindows()
    out.release()
    cap.release()

    print(time.time() - start_time)
    if (time.time() - start_time < duration):
        if os.path.isfile(filename):
            time.sleep(1)
            os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('monitor_id')
    parser.add_argument('duration')
    args = parser.parse_args()
    monitor_id = args.monitor_id
    duration = int(args.duration)
    print(duration)
    # g.logger.Info('my_script_lalala')

    gen(monitor_id, duration)
