#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from six.moves import urllib, range
from six.moves import cPickle as pickle
import tarfile


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth), dtype=np.int32)
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def load_cifar10(path, normalize=True, dequantify=False, one_hot=True):
    """
    Loads the cifar10 dataset.

    :param path: path to dataset file.
    :param normalize: normalize the x data to the range [0, 1].
    :param dequantify: Add uniform noise to dequantify the data following (
        Uria, 2013).
    :param one_hot: Use one-hot representation for the labels.

    :return: The cifar10 dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset(
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path)

    data_dir = os.path.dirname(path)
    batch_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.isfile(os.path.join(batch_dir, 'data_batch_5')):
        with tarfile.open(path) as tar:
            tar.extractall(data_dir)

    train_x, train_y = [], []
    for i in range(1, 6):
        batch_file = os.path.join(batch_dir, 'data_batch_' + str(i))
        with open(batch_file, 'rb') as f:
            data = pickle.load(f,  encoding='latin')
            train_x.append(data['data'])
            train_y.append(data['labels'])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    test_batch_file = os.path.join(batch_dir, 'test_batch')
    with open(test_batch_file, 'rb') as f:
        data = pickle.load(f, encoding='latin')
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    if dequantify:
        train_x += np.random.uniform(0, 1,
                                     size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0, 1, size=test_x.shape).astype('float32')
    if normalize:
        train_x /= 256
        test_x /= 256

    train_x = train_x.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_x = test_x.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return train_x, t_transform(train_y), test_x, t_transform(test_y)








