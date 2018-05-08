#!/usr/bin/env python3

import glob
import argparse
import sys
import hashlib
import io

from os import path, mkdir, remove
from shutil import rmtree
from random import shuffle
from lxml import etree

from funcy import compose
from tqdm import tqdm
from PIL import Image

import tensorflow as tf

from tflib.object_detection.utils import dataset_util
from tflib.object_detection.utils import label_map_util


def get_default_label_map():
    return 'configs/label_map.pbtxt'


def get_default_data_dir():
    return 'temp/data'


def get_default_extract_count():
    return 100


def get_default_split_ratio():
    return 0.1


def create_example(entry, label_map_dict):
    img_path = entry[0]
    label_path = entry[1]
    try:
        with tf.gfile.GFile(label_path, 'r') as fid:
            xml_str = bytes(bytearray(fid.read(), encoding='utf-8'))
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)

        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = image.size
        width = int(width)
        height = int(height)

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        if 'object' in data:
            for obj in data['object']:
                difficult_obj.append(int(0))

                _xmin = max(float(obj['bndbox']['xmin']), 0)
                _ymin = max(float(obj['bndbox']['ymin']), 0)
                _xmax = min(float(obj['bndbox']['xmax']), width)
                _ymax = min(float(obj['bndbox']['ymax']), height)

                xmin.append(_xmin / width)
                ymin.append(_ymin / height)
                xmax.append(_xmax / width)
                ymax.append(_ymax / height)

                class_name = obj['name']
                classes_text.append(class_name.encode('utf8'))
                classes.append(label_map_dict[class_name])
                truncated.append(int(0))
                poses.append('Unspecified'.encode('utf8'))

            return tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    data['filename'].encode('utf8')
                ),
                'image/source_id': dataset_util.bytes_feature(
                    data['filename'].encode('utf8')
                ),
                'image/key/sha256': dataset_util.bytes_feature(
                    key.encode('utf8')
                ),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(
                    'jpeg'.encode('utf8')
                ),
                'image/object/bbox/xmin': dataset_util.float_list_feature(
                    xmin
                ),
                'image/object/bbox/xmax': dataset_util.float_list_feature(
                    xmax
                ),
                'image/object/bbox/ymin': dataset_util.float_list_feature(
                    ymin
                ),
                'image/object/bbox/ymax': dataset_util.float_list_feature(
                    ymax
                ),
                'image/object/class/text': dataset_util.bytes_list_feature(
                    classes_text
                ),
                'image/object/class/label': dataset_util.int64_list_feature(
                    classes
                ),
                'image/object/difficult': dataset_util.int64_list_feature(
                    difficult_obj
                ),
                'image/object/truncated': dataset_util.int64_list_feature(
                    truncated
                ),
                'image/object/view': dataset_util.bytes_list_feature(poses),
            }))
    except ValueError as err:
        print(img_path)
        print(label_path)
        print(err)
        return None


def select(count):
    def get_subset(arr):
        shuffle(arr)
        max_size = min(count, len(arr))
        return arr[:max_size]
    return get_subset


def handle(files):
    arr = []
    for i in range(len(files)):
        imagesrc = str(files[i])
        xml_file = imagesrc.replace('images/', 'labels/')
        xml_file = xml_file.replace('.jpg', '.xml')
        if path.isfile(xml_file):
            arr.append([imagesrc, xml_file])
    return arr


def check(d):
    files = []
    if path.isdir(d):
        files = glob.glob(d + '/images/*.jpg')
    return files


def load(d, count):
    return compose(select(count), handle, check)(d)


def process(entries, output_dir, label_map, split_ratio):
    rat = float(split_ratio)
    if rat >= 1 or rat < 0:
        rat = get_default_split_ratio()
    total = len(entries)
    test_size = round(rat * total)
    training_size = total - test_size
    print('test/train/total {}/{}/{}'.format(test_size, training_size, total))

    test_set = entries[:test_size]
    training_set = entries[test_size:]

    label_map_dict = label_map_util.get_label_map_dict(label_map)
    print(label_map_dict)

    tfwriter = tf.python_io.TFRecordWriter

    print('Handling training set ({})'.format(training_size))
    train_writer = tfwriter(output_dir + '/train.record')
    for entry in tqdm(training_set):
        try:
            exp = create_example(entry, label_map_dict)
            if exp is not None:
                train_writer.write(exp.SerializeToString())
        except ValueError as err:
            print(err)
            continue
    train_writer.close()

    print('Handling test set ({})'.format(test_size))
    test_writer = tfwriter(output_dir + '/eval.record')
    for entry in tqdm(test_set):
        try:
            exp = create_example(entry, label_map_dict)
            if exp is not None:
                test_writer.write(exp.SerializeToString())
        except ValueError as err:
            print(err)
            continue
    test_writer.close()


def preload(input_dir, extracting_count, output_dir, label_map, split_ratio):
    if path.exists(output_dir):
        rmtree(output_dir)
    mkdir(output_dir)
    files = load(input_dir, int(extracting_count))
    total = len(files)

    if total > 0:
        print('Selected {} entries to process'.format(total))
        return process(files, output_dir, label_map, split_ratio)
    else:
        print('No input label & image. Stopped!')


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dir',
        help='Path to dataset. Default "../vgg-faces-utils/output"'
    )
    parser.add_argument(
        '-l',
        '--labelmap',
        help='Path to label map. Default "configs/label_map.pbtxt"'
    )
    parser.add_argument(
        '-e',
        '--extract',
        help='How many items? Default 100'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Path to output dir. Default "temp/data"'
    )
    parser.add_argument(
        '-r',
        '--ratio',
        help='Ratio of Training/Test set. Default 0.1 (9 train/1 eval)'
    )
    args = parser.parse_args()
    if not args.dir:
        print('Please specify path to source dir')
    else:
        label_map = args.labelmap
        if label_map is None:
            label_map = get_default_label_map()

        count = args.extract
        if count is None:
            count = get_default_extract_count()

        odir = args.output
        if odir is None:
            odir = get_default_data_dir()

        ratio = args.ratio
        if ratio is None:
            ratio = get_default_split_ratio()

        entries = preload(
            path.normpath(args.dir),
            count,
            odir,
            label_map,
            ratio
        )


if __name__ == '__main__':
    start()

