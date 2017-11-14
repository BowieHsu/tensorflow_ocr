import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1
from nets import resnet_v2

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
            num_outputs = [2, 2, 2, 2]
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
            geo_map = slim.conv2d(g[3], 16, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            # F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, geo_map 


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
    add OHEM mode
    '''

    classification_loss = dice_coefficient(y_true_pixel, y_pred_pixel, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.02

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, d5_gt, d6_gt, d7_gt, d8_gt= tf.split(value=y_true_link, num_or_size_splits=8, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, d5_pred, d6_pred, d7_pred, d8_pred= tf.split(value=y_pred_link, num_or_size_splits=8, axis=3)

    d1_pos_n = tf.reduce_mean(tf.cast(tf.equal(d1_gt, 1),tf.float32))
    d2_pos_n = tf.reduce_mean(tf.cast(tf.equal(d2_gt, 1),tf.float32))
    d3_pos_n = tf.reduce_mean(tf.cast(tf.equal(d3_gt, 1),tf.float32))
    d4_pos_n = tf.reduce_mean(tf.cast(tf.equal(d4_gt, 1),tf.float32))
    d5_pos_n = tf.reduce_mean(tf.cast(tf.equal(d5_gt, 1),tf.float32))
    d6_pos_n = tf.reduce_mean(tf.cast(tf.equal(d6_gt, 1),tf.float32))
    d7_pos_n = tf.reduce_mean(tf.cast(tf.equal(d7_gt, 1),tf.float32))
    d8_pos_n = tf.reduce_mean(tf.cast(tf.equal(d8_gt, 1),tf.float32))

    d1_link_loss = - tf.reduce_mean(d1_gt * tf.log(d1_pred + 1e-10))/d1_pos_n
    d2_link_loss = - tf.reduce_mean(d1_gt * tf.log(d2_pred + 1e-10))/d2_pos_n
    d3_link_loss = - tf.reduce_mean(d1_gt * tf.log(d3_pred + 1e-10))/d3_pos_n
    d4_link_loss = - tf.reduce_mean(d1_gt * tf.log(d4_pred + 1e-10))/d4_pos_n
    d5_link_loss = - tf.reduce_mean(d1_gt * tf.log(d5_pred + 1e-10))/d5_pos_n
    d6_link_loss = - tf.reduce_mean(d1_gt * tf.log(d6_pred + 1e-10))/d6_pos_n
    d7_link_loss = - tf.reduce_mean(d1_gt * tf.log(d7_pred + 1e-10))/d7_pos_n
    d8_link_loss = - tf.reduce_mean(d1_gt * tf.log(d8_pred + 1e-10))/d8_pos_n

    link_loss = d1_link_loss + d2_link_loss + d3_link_loss + d4_link_loss + d5_link_loss + d6_link_loss + d7_link_loss + d8_link_loss

    # d1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d1_pred,
                         # labels = tf.cast(d1_gt, dtype = tf.int32)))

    # d2_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d2_pred,
                         # labels = tf.cast(tf.squeeze(d2_gt), dtype = tf.int32)))/d2_pos_n
    
    # d3_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d3_pred,
                         # labels = tf.cast(tf.squeeze(d3_gt), dtype = tf.int32)))/d3_pos_n

    # d4_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d4_pred,
                         # labels = tf.cast(tf.squeeze(d4_gt), dtype = tf.int32)))/d4_pos_n

    # d5_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d5_pred,
                         # labels = tf.cast(tf.squeeze(d5_gt), dtype = tf.int32)))/d5_pos_n

    # d6_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d6_pred,
                         # labels = tf.cast(tf.squeeze(d6_gt), dtype = tf.int32)))/d6_pos_n

    # d7_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d7_pred,
                         # labels = tf.cast(tf.squeeze(d7_gt), dtype = tf.int32)))/d7_pos_n

    # d8_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         # logits = d8_pred,
                         # labels = tf.cast(tf.squeeze(d8_gt), dtype = tf.int32)))/d8_pos_n

    # link_loss = d1_loss + d2_loss + d3_loss + d4_loss + d5_loss + d6_loss + d7_loss + d8_loss
    # area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    # area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    # w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    # h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    # area_intersect = w_union * h_union
    # area_union = area_gt + area_pred - area_intersect
    # L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    # L_theta = 1 - tf.cos(theta_pred - theta_gt)
    # tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    # tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    # L_g = L_AABB + 20 * L_theta

    return link_loss + classification_loss
    # return link_loss + pixel_loss 
