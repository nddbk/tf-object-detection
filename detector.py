#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf

from tflib.object_detection.utils import label_map_util


NUM_CLASSES = 1
MIN_THRESHOLD = 0.5

PAINT_COLOR = (0, 0, 255)
PAINT_SIZE = 1

TEXT_COLOR = (0, 0, 255)
TEXT_FONT = cv2.FONT_HERSHEY_PLAIN

FONT_SIZE = 1
FONT_WEIGHT = 1


def draw_boxes(image, boxes, labels, scores):
    k = 0
    for box in boxes:
        label = labels[k]
        score = scores[k]
        score = np.around(score, 2)

        top = box[0]
        left = box[1]
        height = box[2]
        width = box[3]

        xmin = int(left * image.shape[1])
        xmax = int(width * image.shape[1])
        ymin = int(top * image.shape[0])
        ymax = int(height * image.shape[0])

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            PAINT_COLOR,
            PAINT_SIZE
        )
        cv2.putText(
            image,
            label + ' ' + str(score),
            (xmin, ymin - 12),
            TEXT_FONT,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_WEIGHT,
            cv2.LINE_AA
        )
        k += 1

    return image


def create_detector(model_path, label_path):
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=NUM_CLASSES,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        d_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        d_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        d_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_d = detection_graph.get_tensor_by_name('num_detections:0')

    sess = tf.Session(graph=detection_graph)

    def detect(img):
        with detection_graph.as_default():
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [
                    d_boxes,
                    d_scores,
                    d_classes,
                    num_d
                ],
                feed_dict={image_tensor: img_expanded}
            )

            picked = list(filter(lambda x: x > MIN_THRESHOLD, scores[0]))
            picked_size = len(picked)
            _boxes = boxes[0][:picked_size]
            _classes = []
            clsids = classes[0][:picked_size]
            for k in clsids:
                if k in category_index.keys():
                    class_name = category_index[k]['name']
                    _classes.append(class_name)
            _scores = scores[0][:picked_size]
            return _boxes, _classes, _scores

    return detect
