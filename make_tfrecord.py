#!/usr/bin/env python3

import argparse

import sys

from random import shuffle
from os import path, mkdir
from shutil import rmtree

from tqdm import tqdm

import tensorflow as tf

from scripts.utils import load, create_example


def get_default_data_dir():
    return 'data'


def get_default_split_ratio():
    return 0.1


def process(entries, output_dir, split_ratio=None):
    rat = float(split_ratio)
    shuffle(entries)
    if rat >= 1 or rat < 0:
        rat = get_default_split_ratio()
    total = len(entries)
    test_size = round(rat * total)
    training_size = total - test_size
    print('test/train/total {}/{}/{}'.format(test_size, training_size, total))

    tfwriter = tf.python_io.TFRecordWriter

    test_set = entries[:test_size]
    training_set = entries[test_size:]

    print('Handling training set ({})'.format(training_size))
    train_writer = tfwriter(output_dir + '/train.record')
    for entry in tqdm(training_set):
        exp = create_example(entry)
        train_writer.write(exp.SerializeToString())
    train_writer.close()

    print('Handling test set ({})'.format(test_size))
    test_writer = tfwriter(output_dir + '/eval.record')
    for entry in tqdm(test_set):
        exp = create_example(entry)
        test_writer.write(exp.SerializeToString())
    test_writer.close()


def preload(input_dir, output_dir, split_ratio=None):
    if path.exists(output_dir):
        rmtree(output_dir)
    mkdir(output_dir)
    files = load(input_dir)
    return process(files, output_dir, split_ratio)


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dir',
        help='Path to dataset. Default "../vgg-faces-utils/output"'
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
        odir = args.output
        if odir is None:
            odir = get_default_data_dir()

        ratio = args.ratio
        if ratio is None:
            ratio = get_default_split_ratio()

        entries = preload(
            path.normpath(args.dir),
            odir,
            ratio
        )


if __name__ == '__main__':
    start()

