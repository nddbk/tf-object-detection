import glob

from os import path, system
from collections import namedtuple

from random import shuffle
from lxml import etree
from funcy import compose

import tensorflow as tf

Example = namedtuple('Example', [
    'fname',
    'shape',
    'boxes',
])

Shape = namedtuple('Shape', [
    'width',
    'height',
    'depth',
])

Box = namedtuple('Box', [
    'id',
    'name',
    'xmin',
    'xmax',
    'ymin',
    'ymax',
])

classes = [
    'face',
]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(item):
    fname = item.fname

    filename = path.basename(fname)
    filepath = path.abspath(fname)

    imgname = filename.encode('utf8')

    shape = item.shape
    width = shape.width
    height = shape.height
    depth = shape.depth

    boxes = item.boxes

    image_string = ''
    image_format = 'jpeg'
    with tf.gfile.GFile(filepath, 'rb') as fs:
        image_string = fs.read()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for box in boxes:
        boxid = box.id
        name = box.name

        xmin.append(box.xmin / width)
        xmax.append(box.xmax / width)
        ymin.append(box.ymin / height)
        ymax.append(box.ymax / height)

        classes.append(boxid)
        classes_text.append(name.encode('utf8'))

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channels': int64_feature(depth),
        'image/filename': bytes_feature(imgname),
        'image/source_id': bytes_feature(imgname),
        'image/encoded': bytes_feature(image_string),
        'image/format': bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/bbox/text': bytes_list_feature(classes_text),
        'image/object/bbox/label': int64_list_feature(classes),
    }))


def extract(arr):
    examples = []
    for item in arr:
        [img_file, label_file] = item

        with open(label_file, 'r') as f:
            tree = etree.parse(f)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)

            shape = Shape(width, height, depth)

            boxes = []
            objects = root.findall('object')
            for obj in objects:
                name = obj.find('name').text
                boxid = classes.index(name) + 1

                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                xmax = float(bbox.find('xmax').text)
                ymin = float(bbox.find('ymin').text)
                ymax = float(bbox.find('ymax').text)

                box = Box(
                    boxid,
                    name,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )
                boxes.append(box)
            exp = Example(img_file, shape, boxes)
            examples.append(exp)
    return examples


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
    return compose(extract, select(count), handle, check)(d)

