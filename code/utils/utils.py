# coding: utf-8
# Author: gjwei
import numpy
import numpy as np
import torch


# For embedding weights
def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def get_mask(lengths, max_length):
    mask = np.zeros((len(lengths), max_length), dtype=np.float32)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return mask


def adjust_learning_rate(optimizer, lr, decay_rate=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr <= 1e-5:
        return lr
    lr = lr * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



