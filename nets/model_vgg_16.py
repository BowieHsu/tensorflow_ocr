import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1
from nets import resnet_v2
from nets import vgg

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def model_resnet_v1_101(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry

def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            feature_maps = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]

            for i in range(4):
                print('Shape of f_{} {}'.format(i, feature_maps[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [2, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = feature_maps[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], feature_maps[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[3], 8, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # geo_map = slim.conv2d(g[3], 16, 1, activation_fn=None, normalizer_fn=None)
            # angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            # F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, geo_map 

def model_vgg(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = vgg.basenet(images, scope='vgg_16')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            feature_maps = [end_points['fc7'], end_points['conv5_3'],
                 end_points['conv4_3'], end_points['conv3_3'], end_points['conv2_2']]

            pixel_1 = slim.conv2d(end_points['fc7'], 2, 1) + slim.conv2d(end_points['conv5_3'], 2, 1)
            pixel_2 = unpool(pixel_1) + slim.conv2d(end_points['conv4_3'], 2, 1)
            pixel_3 = unpool(pixel_2) + slim.conv2d(end_points['conv3_3'], 2, 1)
            pixel_cls = slim.conv2d(pixel_3, 2, 1)

            print('pixel_shape:{}'.format(pixel_cls.shape))

            link_1 = slim.conv2d(end_points['fc7'], 16, 1) + slim.conv2d(end_points['conv5_3'], 16, 1)
            link_2 = unpool(link_1) + slim.conv2d(end_points['conv4_3'], 16, 1)
            link_3 = unpool(link_2) + slim.conv2d(end_points['conv3_3'], 16, 1)
            link_cls = slim.conv2d(link_3, 16, 1)

            print('link_shape:{}'.format(link_cls.shape))

    return pixel_cls, link_cls

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def loss(y_true_pixel, y_pred_pixel,
         y_true_link, y_pred_link,
         training_mask):
    '''
    return pixel loss and link loss 
    using dice classification loss
    '''

    classification_loss = dice_coefficient(y_true_pixel, y_pred_pixel, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 2

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, d5_gt, d6_gt, d7_gt, d8_gt= tf.split(value=y_true_link, num_or_size_splits=8, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, d5_pred, d6_pred, d7_pred, d8_pred= tf.split(value=y_pred_link, num_or_size_splits=8, axis=3)

    d1_loss = dice_coefficient(d1_gt, d1_pred, training_mask)
    d2_loss = dice_coefficient(d2_gt, d2_pred, training_mask)
    d3_loss = dice_coefficient(d3_gt, d3_pred, training_mask)
    d4_loss = dice_coefficient(d4_gt, d4_pred, training_mask)
    d5_loss = dice_coefficient(d5_gt, d5_pred, training_mask)
    d6_loss = dice_coefficient(d6_gt, d6_pred, training_mask)
    d7_loss = dice_coefficient(d7_gt, d7_pred, training_mask)
    d8_loss = dice_coefficient(d8_gt, d8_pred, training_mask)

    link_loss = d1_loss + d2_loss + d3_loss + d4_loss + d5_loss + d6_loss + d7_loss + d8_loss

    tf.summary.scalar('link_loss', link_loss)

    return link_loss + classification_loss

def cal_link_loss(link_gt, link_pred, W_pixel):
    '''
    cal link loss on up down left right left_up left_down right_up right_down
    '''
    link_gt = tf.cast(tf.reshape(link_gt, [-1]),tf.int32)
    link_pred = tf.reshape(link_pred, [-1, 2])

    W_pos_link = tf.cast(tf.equal(link_gt, 1), tf.float32) * W_pixel
    W_neg_link = tf.cast(tf.equal(link_gt, 0), tf.float32) * W_pixel

    L_link_pos = tf.reduce_sum((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = link_pred, labels = link_gt)) * W_pos_link)/tf.reduce_sum(W_pos_link)

    L_link_neg = tf.reduce_sum((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = link_pred, labels = link_gt)) * W_neg_link)/tf.reduce_sum(W_neg_link)
    
    return L_link_pos + L_link_neg

def ohem_loss(y_true_pixel, y_pred_pixel,
              y_true_link, y_pred_link,
              training_mask):

    '''
    without ohem, calculate loss funcation

    L_pixel    = W * L_pixel_CE
    W_pos_link = W * arg(y_true_link == 1)
    W_neg_link = W * arg(y_true_link == 0)
    L_link_pos = W_pos_link * L_link_CE
    L_link_neg = W_neg_link * L_link_CE
    L_link = L_link_pos/sum(W_pos_link) + L_link_neg/sum(W_neg_link)
    Loss = L_pixel * 2 + L_link
    '''

    y_true_pixel = tf.cast(tf.reshape(y_true_pixel, [-1]), tf.int32)
    y_pred_pixel = tf.reshape(y_pred_pixel, [-1, 2])

    #cal pixel weight
    W_pixel = tf.cast(tf.equal(y_true_pixel, 1), tf.float32)

    L_pixel = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_pred_pixel, labels = y_true_pixel) * W_pixel)/tf.reduce_sum(W_pixel)

    #cal link 
    d1_gt, d2_gt, d3_gt, d4_gt, d5_gt, d6_gt, d7_gt, d8_gt= tf.split(value=y_true_link, num_or_size_splits=8, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, d5_pred, d6_pred, d7_pred, d8_pred= tf.split(value=y_pred_link, num_or_size_splits=8, axis=3)

    d1_loss = cal_link_loss(d1_gt, d1_pred, W_pixel)
    d2_loss = cal_link_loss(d2_gt, d2_pred, W_pixel)
    d3_loss = cal_link_loss(d3_gt, d3_pred, W_pixel)
    d4_loss = cal_link_loss(d4_gt, d4_pred, W_pixel)
    d5_loss = cal_link_loss(d5_gt, d5_pred, W_pixel)
    d6_loss = cal_link_loss(d6_gt, d6_pred, W_pixel)
    d7_loss = cal_link_loss(d7_gt, d7_pred, W_pixel)
    d8_loss = cal_link_loss(d8_gt, d8_pred, W_pixel)

    L_link = d1_loss + d2_loss + d3_loss + d4_loss + d5_loss + d6_loss + d7_loss + d8_loss

    return L_pixel * 2 + L_link
