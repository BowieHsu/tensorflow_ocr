#
# Authors:Bowen Xu
# =============================================================================
import tensorflow as tf
import numpy as np
import tf.slim as slim

def ocr_net():
    with tf.name_scope('OCR_NET'):

        with tf.name_scope('CONV_1'):
            net = slim.conv2d(net, 64, [3, 3], scope='conv1')
