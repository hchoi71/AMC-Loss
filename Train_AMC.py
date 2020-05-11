
"""
Quick Start AMC model for CIFAR10.
"""

# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import math
import numpy as np
from six.moves import range
import tensorflow as tf
from functools import wraps
import re

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers import batch_norm, fully_connected, conv2d, \
    max_pool2d, dropout

import dataset
from utils import lrelu, GaussianNoise, report_parameters

save_dir = os.environ['MODEL_RESULT_PATH_AND_PREFIX'] \
    if 'MODEL_RESULT_PATH_AND_PREFIX' in os.environ else '.'


tf.flags.DEFINE_string('save_dir', save_dir, 'Location for parameter checkpoints and samples')
tf.flags.DEFINE_integer('seed', 1, "Random seed to use")
tf.flags.DEFINE_float('polyak_decay',           0.999, "Exponential decay rate of the sum of previous model iterates during Polyak averaging")
tf.flags.DEFINE_float('adam_beta1',             0.9, "adam optimizer initial beta1")
tf.flags.DEFINE_integer('augment_translation',  2, "")
tf.flags.DEFINE_float('augment_noise_stddev',   0.15, "")
tf.flags.DEFINE_bool('augment_mirror',          False, "True or False")
tf.flags.DEFINE_bool('whiten_input',            False, "True or False")
tf.flags.DEFINE_float('corruption_percentage',  0, "")
tf.flags.DEFINE_float('coeff_gs',               0.1, "coefficient gs")
tf.flags.DEFINE_float('geo_margin',             0.5, "geo_margin")
tf.flags.DEFINE_float('feaure_scaling',         1.0, "feature_scaling")
tf.flags.DEFINE_bool('AMC',                     False, "True or False")
tf.flags.DEFINE_float('learning_rate',          0.003, "")
tf.flags.DEFINE_integer('max_unl_per_epoch',    None, "")
tf.flags.DEFINE_integer('num_epochs',           200, "")
tf.flags.DEFINE_integer('rampup_length',        50, "")
tf.flags.DEFINE_integer('rampdown_length',      30, "")
tf.flags.DEFINE_float('rd_beta1_trgt',          0.5, "Ramp Adam beta1 down during last n epochs.")
tf.flags.DEFINE_integer('start_epoch',          0, "which epoch to start training")
tf.flags.DEFINE_float('wght_max',               100.0, "")

flgs = tf.flags.FLAGS  # Define training/evaluation parameters

bs = 128
tt_bs = 100
print_freq = 1
tsne_freq = 100


def reuse(scope):
    """
    A decorator for transparent reuse of `tf.Variable` s in a function.
    When a `StochasticGraph` is reused as in a function, this decorator helps
    reuse the `tf.Variable` s in the graph every time the function is called.

    :param scope: A string. The scope name passed to `tf.variable_scope()`.
    """

    def reuse_decorator(f):
        @wraps(f)
        def _func(*args, **kwargs):
            try:
                with tf.variable_scope(scope, reuse=True):
                    return f(*args, **kwargs)
            except ValueError as e:
                if re.search(r'.*not exist.*tf\.get_variable.*', str(e)):
                    with tf.variable_scope(scope):
                        return f(*args, **kwargs)
                else:
                    raise

        return _func

    return reuse_decorator


@reuse('network')
def Model(x, is_training, init=False, ema=None):
    norm_prms = {'is_training': is_training}
    with arg_scope([conv2d], normalizer_fn=batch_norm,
                   activation_fn=lrelu,
                   normalizer_params=norm_prms): 
        ly_x = tf.reshape(x, [-1, 32, 32, 3])
        ly_x = GaussianNoise(ly_x, sigma=flgs.augment_noise_stddev,
                             is_training=is_training)
        ly_x = conv2d(ly_x, 64, 3)
        ly_x = conv2d(ly_x, 64, 3)
        ly_x = conv2d(ly_x, 64, 3)
        ly_x = max_pool2d(ly_x, kernel_size=2)
        ly_x = dropout(ly_x, keep_prob=0.5, is_training=is_training)
        ly_x = conv2d(ly_x, 128, 3)
        ly_x = conv2d(ly_x, 128, 3)
        ly_x = conv2d(ly_x, 128, 3)
        ly_x = max_pool2d(ly_x, kernel_size=2)
        ly_x = dropout(ly_x, keep_prob=0.5, is_training=is_training)
        ly_x = conv2d(ly_x, 256, 3, padding='VALID')
        ly_x_top = conv2d(ly_x, 128, 1)
        ly_x = tf.reduce_mean(ly_x_top, axis=[1, 2])
        class_logits = fully_connected(ly_x, 10, activation_fn=None)
    return class_logits, ly_x, ly_x_top


def rampup(epoch):
    if epoch < flgs.rampup_length:
        p = max(0.0, float(epoch)) / float(flgs.rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0


def rampdown(epoch):
    if epoch >= (flgs.num_epochs - flgs.rampdown_length):
        ep = (epoch - (flgs.num_epochs - flgs.rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / flgs.rampdown_length)
    else:
        return 1.0

def whiten_norm(x):
    x -= np.mean(x, axis=(1, 2, 3), keepdims=True)
    x /= np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5
    return x

def prepare_dataset(result_subdir, x_train, y_train, x_test, y_test,
                    num_classes):
    # Whiten input data
    if flgs.whiten_input is True:
        x_train = whiten_norm(x_train)
        x_test = whiten_norm(x_test)
    

    # Pad according to the amount of jitter we plan to have.
    p = flgs.augment_translation
    if p > 0:
        x_train = np.pad(x_train, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

    # Random shuffle.
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    print(x_train.shape, y_train.shape, x_test.shape)
    mask_train = np.ones(len(y_train), dtype=np.float32)
    print("Keeping all labels.")

    # Zero out masked-out labels for maximum paranoia.
    for i in range(len(y_train)):
        if mask_train[i] != 1.0:
            y_train[i] = 0
    return x_train, y_train, mask_train, x_test, y_test


def iterate_minibatches(inputs, labels, mask, batch_size):
    assert len(inputs) == len(labels) == len(mask)
    crop = flgs.augment_translation

    num = len(inputs)
    if flgs.max_unl_per_epoch is None:
        indices = np.arange(num)
    else:
        labeled_indices = [i for i in range(num) if mask[i] > 0]
        unlabeled_indices = [i for i in range(num) if mask[i] == 0.0]
        np.random.shuffle(unlabeled_indices)
        indices = labeled_indices + unlabeled_indices[:flgs.max_unl_per_epoch]
        indices = np.asarray(indices)
        num = len(indices)

    np.random.shuffle(indices)

    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            excerpt = indices[start_idx: start_idx + batch_size]
            noisy_a = []
            for img in inputs[excerpt]:
                if flgs.augment_mirror and np.random.uniform() > 0.5:
                    img = img[:, ::-1, :]
                ofs0 = np.random.randint(0, 2 * crop + 1)
                ofs1 = np.random.randint(0, 2 * crop + 1)
                noisy_a.append(img[ofs0:ofs0 + 32, ofs1:ofs1 + 32, :])
            noisy_a = np.array(noisy_a)
            yield noisy_a, labels[excerpt], mask[excerpt]


def evaluation(ema, train_var, eval_var):
    updates = []
    for (var, var_eval) in zip(train_var, eval_var):
        var_avg = ema.average(var)
        updates.append(var_eval.assign(var_avg))
    return tf.group(*updates)


if __name__ == "__main__":

    # fix random seed for reproducibility
    np.random.seed(flgs.seed)
    tf.set_random_seed(flgs.seed)


    data_path = os.path.join('./data/cifar10','cifar-10-python.tar.gz')
    x_train, y_train, x_test, y_test = dataset.load_cifar10(data_path, normalize=True, one_hot=False)
    num_classes = len(set(y_train))
    n_data, n_xl, _, n_channels = x_train.shape
    n_x = n_xl * n_xl * n_channels
    
    # prepare data
    x_train, y_train, mask_train, x_test, y_test = prepare_dataset(flgs.save_dir, x_train, y_train, x_test, y_test, num_classes)

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    adam_beta1_ph = tf.placeholder(tf.float32, shape=[], name='beta1')
    weight_ph = tf.placeholder(tf.float32, shape=[], name='wght')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=adam_beta1_ph)

    # data placeholders
    x_init = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels), name='x_init')
    x_ph = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels), name='x_1')
    y_ph = tf.placeholder(tf.int32, shape=(None,), name='y')
    mask_ph = tf.placeholder(tf.float32, shape=(None,), name='mask')
    target_pred = tf.placeholder(tf.float32, shape=(None, num_classes), name='targets_te')

    
    # Outputs
    d_logits1, d_emb1, top_conv = Model(x_ph, is_training)
    target_pred = tf.nn.softmax(d_logits1)

    # costs, for tempens model d_logits1 is target logits of previous ensemble
    print(' =========  Cross-entropy_loss  ======== ')
    cross_ent = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_ph, logits=d_logits1) * mask_ph)
    
    cost = cross_ent
    
    geo_loss = tf.constant(0.)        
    if flgs.AMC is True:
        print(' =========  Adding AMC_loss  ========= ')
        print('Geo Coeff      : {}'.format(flgs.coeff_gs))
        print('geo_margin     : {}'.format(flgs.geo_margin))

        half = tf.to_int32(tf.to_float(tf.shape(d_emb1)[0]) / 2.)
        target_hard = tf.to_int32(tf.argmax(target_pred, axis=1))
        merged_tar = tf.where(tf.equal(mask_ph, 1), target_hard, y_ph)
        neighbor_bool = tf.equal(merged_tar[:half], merged_tar[half:])

        phi_emb = tf.nn.l2_normalize(d_emb1, axis=1)
        inner = tf.reduce_sum(phi_emb[:half]*phi_emb[half:], axis=1) # Geo_Loss_v1
        geo_desic = tf.acos(tf.clip_by_value(inner, -1.0+1e-07, 1.0-1e-07)) * flgs.feaure_scaling
        geo_losses = tf.where(neighbor_bool, tf.square(geo_desic), tf.square(tf.maximum(flgs.geo_margin - geo_desic, 0)))  # version 1
        geo_loss = tf.reduce_mean(geo_losses, name="loss")
        cost += weight_ph * geo_loss * flgs.coeff_gs
        
    # get variable lists
    disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='network')
    disc_global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope='network')

    # EMA
    with tf.variable_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=flgs.polyak_decay)
        maintain_averages_op = ema.apply(disc_global_vars)

    # batch norm updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        infer = optimizer.minimize(cost, var_list=disc_var_list)

    with tf.control_dependencies([infer, maintain_averages_op]):
        train_op = tf.no_op('train')

    # evaluation
    with tf.variable_scope('eval'):
        eval_d_logits_l, eval_h_embed, _ = Model(x_ph, is_training)
        test_acc = tf.reduce_mean(tf.cast(tf.equal(
            tf.cast(tf.argmax(eval_d_logits_l, 1), tf.int32), y_ph),
        tf.float32))
        test_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=eval_d_logits_l))
    eval_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='eval')
    eval_op = evaluation(ema, disc_global_vars, eval_vars)


    params = tf.trainable_variables()
    report_parameters(params)

    saver = tf.train.Saver()

    # Run the inference
    with tf.Session() as sess:
        p = flgs.augment_translation
        sess.run(tf.global_variables_initializer(),
                 feed_dict={x_init: x_train[:200, p: p+n_xl, p: p+n_xl, :],
                            is_training: False,
                            adam_beta1_ph: flgs.adam_beta1
                            })
        print('Train the model...')
        
        for epoch in range(flgs.start_epoch, flgs.num_epochs):
            ru = rampup(epoch)
            rd = rampdown(epoch)
            lr = ru * rd * flgs.learning_rate
            adam_beta1 = rd * flgs.adam_beta1 + (1.0 - rd) * flgs.rd_beta1_trgt
            scale_wght_max = flgs.wght_max
            lambda_weight = ru * scale_wght_max
            if epoch == flgs.start_epoch:
                lambda_weight = 0.0

            fetches = []
            time_train = -time.time()
            minibatches = iterate_minibatches(x_train, y_train, mask_train, bs)
            for (x_btc_a, y_btc, mask_btc) in minibatches:
                ft = sess.run(
                    [train_op, cost, cross_ent, geo_loss],
                    feed_dict={x_ph: x_btc_a,
                               y_ph: y_btc,
                               mask_ph: mask_btc,
                               learning_rate_ph: lr,
                               weight_ph: lambda_weight,
                               adam_beta1_ph: adam_beta1,
                               is_training: True})
                fetches.append(ft[1:])
            ## Evaluate Training
            pt_fetches = np.mean(fetches, axis=0)
            print('Epoch={} ({:.3f}s):' 'Loss = {:.5f} cross_ent loss = {:.5f}, geo_loss = {:.5f}'.format(epoch+1, (time.time() + time_train), *pt_fetches))
            
            if epoch % print_freq == 0:
                  
                fetches = []
                time_train = -time.time()
                
                # Evaluation Test
                time_test = -time.time()
                tt_accs = []
                tt_embs = []
                for tt in range(x_test.shape[0] // tt_bs):
                    test_x_batch = x_test[tt * tt_bs: (tt + 1) * tt_bs]
                    test_y_batch = y_test[tt * tt_bs: (tt + 1) * tt_bs]
                    sess.run(eval_op)
                    if (epoch+1) % tsne_freq == 0:
                        test_fetches = [test_acc, test_cost, eval_h_embed]
                    else:
                        test_fetches = [test_acc, test_cost]
                    tst_v = sess.run(test_fetches,
                                     feed_dict={
                                         x_ph: test_x_batch,
                                         y_ph: test_y_batch,
                                         is_training: False})
                    t_acc = tst_v[0]
                    tt_accs.append(t_acc)
                    if (epoch+1) % tsne_freq == 0:
                        t_emb = tst_v[2]
                        tt_embs.append(t_emb)

                time_test += time.time()
                tt_acc = 100.*np.mean(tt_accs)
                print('>>> TEST EPOCH {} ({:.1f}s)'.format(epoch+1, time_test))
                print('>> Test accuracy: {:.2f}%'.format(tt_acc))
