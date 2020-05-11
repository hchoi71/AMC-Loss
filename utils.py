#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division


from contextlib import contextmanager
import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.contrib.framework.python.ops import add_arg_scope
import numpy as np
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
from six.moves import range
import os
import sys
import datetime
import socket
import getpass
import glob
import shutil
import time


class AdamaxOptimizer(optimizer.Optimizer):
    """
    Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: uint8 numpy array
        The output image.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def save_contrast_image_collections(x1, x2, filename, shape=(10, 10),
                                    scale_each=False, transpose=False,
                                    along_col=True):
    """
    :param x1: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param x2: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param shape: tuple
        The shape of final big images.
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :param along_col: bool
        If true, the contrastive images are placed one by one along the column.
        If False, they are placed one by one along the row.
    :return: uint8 numpy array
        The output image.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x1.shape[0]
    if transpose:
        x1 = x1.transpose(0, 2, 3, 1)
        x2 = x2.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x1[i] = rescale_intensity(x1[i], out_range=(0, 1))
            x2[i] = rescale_intensity(x2[i], out_range=(0, 1))
    n_channels = x1.shape[3]
    x1 = img_as_ubyte(x1)
    x2 = img_as_ubyte(x2)
    r, c = shape
    if r * c < 2 * n:
        print('Shape too small to contain all images')
    h, w = x1.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < 2 * n:
                if along_col:
                    if j % 2 == 0:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x1[
                            int(i * c / 2 + j / 2)]
                    else:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x2[
                            int(i * c / 2 + (j - 1) / 2)]
                else:
                    if i % 2 == 0:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x1[
                            int(j * r / 2 + i / 2)]
                    else:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x2[
                            int(j * r / 2 + (i - 1) / 2)]

    ret = ret.squeeze()
    io.imsave(filename, ret)


@contextmanager
def name_variable_scope(name_scope_name, var_scope_or_var_scope_name,
                        *var_scope_args, **var_scope_kwargs):
    """A combination of name_scope and variable_scope with different names

    The tf.variable_scope function creates both a name_scope and a variable_scope
    with identical names. But the naming would often be clearer if the names
    of operations didn't inherit the scope name of the (reused) variables.
    So use this function to make shorter and more logical scope names in these cases.
    """
    with tf.name_scope(name_scope_name) as outer_name_scope:
        with tf.variable_scope(var_scope_or_var_scope_name,
                               *var_scope_args,
                               **var_scope_kwargs) as var_scope:
            with tf.name_scope(outer_name_scope) as inner_name_scope:
                yield inner_name_scope, var_scope


@contextmanager
def ema_variable_scope(name_scope_name, var_scope, decay=0.999):
    """Scope that replaces trainable variables with their exponential moving averages

    We capture only trainable variables. There's no reason we couldn't support
    other types of variables, but the assumed use case is for trainable variables.
    """
    with tf.name_scope(name_scope_name + "/ema_variables"):
        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name)
        }
        ema = tf.train.ExponentialMovingAverage(decay)
        update_op = ema.apply(original_trainable_vars.values())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    def use_ema_variables(getter, name, *_, **__):
        #pylint: disable=unused-argument
        assert name in original_trainable_vars, "Unknown variable {}.".format(name)
        return ema.average(original_trainable_vars[name])

    with name_variable_scope(name_scope_name,
                             var_scope,
                             custom_getter=use_ema_variables) as (name_scope, var_scope):
        yield name_scope, var_scope


def lrelu(input_tensor, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input_tensor + f2 * abs(input_tensor)


@add_arg_scope
def GaussianNoise(x, sigma, is_training, name=None):
    with tf.name_scope(name, 'gaussian_noise', [x, sigma, is_training]) as scope:
        if sigma == 0.:
            return x
        else:
            x = tf.cond(
                is_training,
                lambda: x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma),
                lambda: x, name=scope)
            return x


def max_out(input_tensor, num_pieces=2):
    shape = input_tensor.get_shape()
    output = tf.reshape(input_tensor, [-1, int(shape[1]), int(shape[2]),
                                       int(shape[3]) // num_pieces, num_pieces])
    output = tf.reduce_max(output, axis=-1)
    return output


def conv_concat(x, y, y_dim=10):
    y = tf.reshape(y, [-1, 1, 1, y_dim])
    x_shape = tf.shape(x)
    return tf.concat([x, y * tf.ones([x_shape[0], x_shape[1], x_shape[2],
                                      y_dim])], axis=3)


def export_run_details(fname):
    host = socket.gethostname().lower()
    user = getpass.getuser()
    with open(fname, 'wt') as f:
        f.write('%-16s%s\n' % ('Host', host))
        f.write('%-16s%s\n' % ('User', user))
        f.write('%-16s%s\n' % ('Date', datetime.datetime.today()))
        f.write('%-16s%s\n' % (
        'CUDA device', os.environ['CUDA_VISIBLE_DEVICES']))
        f.write('%-16s%s\n' % ('Working dir', os.getcwd()))
        f.write('%-16s%s\n' % ('Executable', sys.argv[0]))
        f.write('%-16s%s\n' % ('Arguments', ' '.join(sys.argv[1:])))


def export_config(config, fname):
    with open(fname, 'wt') as fout:
        for k, v in sorted(config.__dict__.iteritems()):
            if not k.startswith('_'):
                fout.write("%s = %s\n" % (k, str(v)))


def export_sources(target_dir):
    os.makedirs(target_dir)
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)


def build_log_file(filename):
    """
    :param filename: Can be os.path.realpath(__file__)
    :return: 
    """
    filename_script = os.path.basename(filename)
    result_out = os.path.join("results", os.path.splitext(filename_script)[0])
    this_time = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    result_out = os.path.join(result_out, os.path.splitext(filename_script)[0]+this_time)
    if not os.path.exists(result_out):
        os.makedirs(result_out)
    shutil.copy(filename, os.path.join(result_out, filename_script))
    logname = 'logfile_'+this_time+'.log'
    logfile = os.path.join(result_out, logname)
    return result_out, this_time, logfile


def report_parameters(params):
    total_params = 0
    for i in params:
        this_shape = i.get_shape()
        print('{} shape: {}'.format(i.name, this_shape.as_list()))
        total_params += int(np.prod(this_shape))
    print('Total params: {}'.format(total_params))


###############################################################################
# Helper class for forking stdout/stderr into a file.
###############################################################################


class Tap:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        self.file = None
        pass

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        if self.file is not None:
            self.file.write(s)
            self.file.flush()
        else:
            self.buffer = self.buffer + s

    def set_file(self, f):
        assert(self.file is None)
        self.file = f
        self.file.write(self.buffer)
        self.file.flush()
        self.buffer = ''

    def flush(self):
        self.stream.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.stream.close()
        if self.file is not None:
            self.file.close()
            self.file = None
