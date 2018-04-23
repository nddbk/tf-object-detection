#!/usr/bin/env python3

import glob
import argparse

import numpy as np
import cv2

from os import path, mkdir
from shutil import rmtree
from random import choice

from tqdm import tqdm

from detector import create_detector, draw_boxes


def get_default_result_dir():
    return 'temp/result'


def get_default_sample_dir():
    return 'samples'


def _predict(image, detect):
    boxes, labels, scores = detect(image)
    # print('boxes:')
    # print(boxes)
    # print('labels:')
    # print(labels)
    # print('scores:')
    # print(scores)
    return draw_boxes(image, boxes, labels, scores)


def predict_image(image_path, detector):
    image = cv2.imread(image_path)
    image = _predict(image, detector)

    while True:
        k = cv2.waitKey(30)
        if k == 27:
            break
        cv2.imshow('Image prediction', image)
    cv2.destroyAllWindows()


def predict_multi(images, output, detector):
    print('Founded {} images. Start handling...'.format(len(images)))
    for img_path in tqdm(images):
        image = cv2.imread(img_path)
        image = _predict(image, detector)
        fname = path.basename(img_path)
        f = output + '/' + fname
        print('Finish handling "{}"'.format(fname))
        cv2.imwrite(f, image)


def predict_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Fail to load video "{}" file'.format(video_path))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        k = cv2.waitKey(30)
        if k == 27:
            break
        frame = _predict(frame, detector)
        cv2.imshow('Video prediction', frame)
    cap.release()
    cv2.destroyAllWindows()


def check(detector, f=None, o=None):
    if isinstance(f, int):
        return predict_video(f, detector)

    if f is None:
        images = glob.glob(get_default_sample_dir() + '/*.jpg')
        f = choice(images)

    if not path.exists(f):
        return print('File/folder not found: "{}"'.format(f))

    if path.isfile(f):
        ext = path.splitext(f)[1]
        if ext in ['.avi', '.mp4', '.mkv']:
            return predict_video(f, detector)
        return predict_image(f, detector)
    if path.isdir(f):
        if o is None:
            o = get_default_result_dir()
        if path.exists(o):
            rmtree(o)
        mkdir(o)
        images = glob.glob(f + '/*.jpg')
        return predict_multi(images, o, detector)


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Relative path to specific model (checkpoint)'
    )
    parser.add_argument(
        '-l',
        '--labelmap',
        help='Relative path to label map'
    )
    parser.add_argument(
        '-f',
        '--file',
        help='Image file to predict'
    )
    parser.add_argument(
        '-c',
        '--cam',
        help='Camera source to predict'
    )
    parser.add_argument(
        '-d',
        '--dir',
        help='Image dir to predict'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Image dir to export the output'
    )

    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        return print('Please specify model to load!')
    label_map = args.labelmap
    if label_map is None:
        return print('Please specify path to label map file!')
    detect_fn = create_detector(model_path, label_map)

    if args.cam:
        check(detect_fn, int(args.cam))
    elif args.dir:
        check(
            detect_fn,
            path.normpath(args.dir),
            path.normpath(args.output),
        )
    elif args.file:
        check(detect_fn, path.normpath(args.file))
    else:
        check(detect_fn)


if __name__ == '__main__':
    start()

