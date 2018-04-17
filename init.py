#!/usr/bin/env python3

from os import path, mkdir

dirs = [
    'temp',
    'temp/checkpoints',
    'temp/data',
    'temp/models',
]


def make(ds):
    for d in ds:
        if not path.exists(d):
            mkdir(d)


if __name__ == '__main__':
    make(dirs)
