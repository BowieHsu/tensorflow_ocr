import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import vgg
import config
import pdb

class PixelLinkNet(object):
    def __init__(self, inputs, weight_decay = None, basenet_type = 'vgg', data_format = 'NHWC',
                 weights_initializer = None, biases_initializer = None):
        self.inputs = inputs
        self.weight_decay = weight_decay
        self.basenet_type = basenet_type
        self.data_format = data_format

        if weights_initializer is None:
            weights_initializer = tf.contrib.layers.xavier_initializer()
        if biases_initializer is None:
            biases_initializer = tf.zeros_initializer()

        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

        self._build_network()
        self.shapes = self.get_shapes()

    def get_shapes(self):
        shapes = {}

        for layer in self.end_points:
            shapes[layer] = tensor_shape(self.end_points[layer])[1:-1]
        return shapes

    def get_shape(self, name):
        return self.shapes[name]

    def unpool(self,inputs):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

    def _build_network(self):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=self.weights_initializer,
                            biases_initializer=self.biases_initializer):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME', data_format = self.data_format):
                with tf.variable_scope(self.basenet_type):
                    # basenet, end_points = net_facatory.get_basenet(self.basenet_type, self.inputs)
                    basenet, end_points = vgg.basenet(self.inputs)
                    self.basenet = basenet
                    self.end_points = end_points

                with tf.variable_scope('pixellink_layers'):
                    self._add_pixellink_layers(basenet, end_points)

    def _add_pixellink_layers(self, basenet, end_points):
        with slim.arg_scope([slim.conv2d], activation_fn=None, weights_regularizer=slim.l2_regularizer(self.weight_decay),weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer = tf.zeros_initializer()):
            pixel_cls_pred = 2
            pixel_cls_stage_1 = slim.conv2d(end_points['fc7'], pixel_cls_pred, [1, 1], scope='stage_6_pixel_fuse') + slim.conv2d(end_points['conv5_3'], pixel_cls_pred, [1, 1], scope='stage_5_pixel_fuse')
            pixel_cls_stage_2 = self.unpool(pixel_cls_stage_1) + slim.conv2d(end_points['conv4_3'], pixel_cls_pred, [1, 1], scope='stage_4_pixel_fuse')
            pixel_cls_stage_3 = self.unpool(pixel_cls_stage_2) + slim.conv2d(end_points['conv3_3'], pixel_cls_pred, [1, 1], scope='stage_3_pixel_fuse')
            pixel_cls = slim.conv2d(pixel_cls_stage_3, pixel_cls_pred, [1, 1], scope='text_predication')

            link_cls_pred = 16
            link_cls_stage_1 = slim.conv2d(end_points['fc7'], link_cls_pred, [1, 1], scope='stage_6_link_fuse') + slim.conv2d(end_points['conv5_3'], link_cls_pred, [1, 1], scope='stage_5_link_fuse')
            link_cls_stage_2 = self.unpool(link_cls_stage_1) + slim.conv2d(end_points['conv4_3'], link_cls_pred, [1, 1], scope='stage_4_link_fuse')
            link_cls_stage_3 = self.unpool(link_cls_stage_2) + slim.conv2d(end_points['conv3_3'], link_cls_pred, [1, 1], scope='stage_3_link_fuse')
            link_cls = slim.conv2d(link_cls_stage_3, link_cls_pred, [1, 1], scope='link_predication')

        self.pixel_cls = pixel_cls
        self.link_cls = link_cls
        self.pixel_scores = slim.softmax(pixel_cls)
        link_scores = slim.softmax(link_cls[:,:,:,0:2])

        pixel_pred_image = tf.expand_dims(self.pixel_scores[0,:,:,1], 0)
        pixel_pred_image = tf.expand_dims(pixel_pred_image, 3)
        tf.summary.image('pixel_pred_image', pixel_pred_image)

        link_pred_image = tf.expand_dims(link_scores[0,:,:,1], 0)
        link_pred_image = tf.expand_dims(link_pred_image, 3)
        tf.summary.image('link_pred_image', link_pred_image)

        # self.link_scores = tf.stack([link_scores[:,:,:,0],link_scores[:,:,:,2], link_scores[:,:,:,4], link_scores[:,:,:,6], link_scores[:,:,:,8], link_scores[:,:,:,10], link_scores[:,:,:,12], link_scores[:,:,:,14]], axis=3)

        tf.summary.histogram('pixel_scores', self.pixel_scores)
        tf.summary.histogram('link_scores', link_scores)
        return pixel_cls, link_cls

    def build_loss(self, pixel_labels, link_labels, do_summary = True):
        batch_size = config.batch_size_per_gpu
        
        # note that for label values in both seg_labels and link_labels:
        #    -1 stands for negative
        #     1 stands for positive
        #     0 stands for ignored
        def get_pos_and_neg_masks(labels):
            # if config.train_with_ignored:
            # pdb.set_trace()
            pos_mask = labels > 0.0
            neg_mask = tf.logical_not(pos_mask)
            # else:
                # pos_mask = tf.equal(labels, 1)
                # neg_mask = tf.equal(labels, -1)
            
            return pos_mask, neg_mask
        
        def OHNM_single_image(scores, n_pos, neg_mask):
            """Online Hard Negative Mining.
                scores: the scores of being predicted as negative cls
                n_pos: the number of positive samples 
                neg_mask: mask of negative samples
                Return:
                    the mask of selected negative samples.
                    if n_pos == 0, no negative samples will be selected.
            """
            def has_pos():
                n_neg = n_pos * config.max_neg_pos_ratio
                boolean_neg_mask = tf.reshape(neg_mask, [-1])
                max_neg_entries = tf.maximum(tf.reduce_sum(tf.cast(boolean_neg_mask, tf.int32)), 1)
                n_neg = tf.minimum(n_neg, max_neg_entries)
                n_neg = tf.cast(n_neg, tf.int32)
                boolean_scores = tf.reshape(scores, [-1])
                float_neg_mask = tf.cast(boolean_neg_mask, tf.float32)
                # neg_conf = tf.boolean_mask(scores, neg_mask)
                neg_conf = tf.where(boolean_neg_mask, boolean_scores, tf.zeros_like(boolean_scores))
                vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
                threshold = vals[-1]# a negtive value
                tf.summary.scalar('threshold', threshold)
                tf.summary.scalar('max_neg_entries', max_neg_entries)
                # pdb.set_trace()
                selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
                return tf.cast(selected_neg_mask, tf.float32)
                
            def no_pos():
                return tf.zeros_like(neg_mask, tf.float32)
            
            return tf.cond(n_pos > 0, has_pos, no_pos) 
        
        def OHNM_batch(neg_conf, pos_mask, neg_mask, weight_mask):
            selected_neg_mask = []
            for image_idx in xrange(batch_size):
                image_neg_conf = neg_conf[image_idx, :, :]
                image_neg_mask = neg_mask[image_idx, :, :]
                image_pos_mask = pos_mask[image_idx, :, :]
                n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
                selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))
                
            selected_neg_mask = tf.stack(selected_neg_mask)
            # selected_mask = tf.cast(weight_mask, tf.float32) + selected_neg_mask
            selected_mask = tf.cast(pos_mask, tf.float32) + selected_neg_mask
            return selected_mask

        # OHNM on segments
        seg_neg_scores = self.pixel_scores[:,:, :, 0]
        seg_pos_mask, seg_neg_mask = get_pos_and_neg_masks(pixel_labels)
        seg_selected_mask = OHNM_batch(seg_neg_scores, seg_pos_mask, seg_neg_mask, pixel_labels)
        n_seg_pos = tf.cast(tf.reduce_sum(seg_selected_mask), tf.float32)
        tf.summary.scalar('n_seg_pos', n_seg_pos)
        
        with tf.name_scope('pixel_cls_loss'):            
            pixel_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pixel_cls, labels = tf.cast(seg_pos_mask, dtype = tf.int32)))
            # def has_pos():
                # seg_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    # logits = self.pixel_cls, 
                    # labels = tf.cast(seg_pos_mask, dtype = tf.int32))
                # return tf.reduce_sum(seg_cls_loss * seg_selected_mask) / n_seg_pos
            # def no_pos():
                # return tf.constant(.0);
            # pixel_cls_loss = tf.cond(n_seg_pos > 0.0, has_pos, no_pos)

            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * 2)
            image_pixel_label = tf.cast(seg_pos_mask[0,:,:], tf.float32)
            image_pixel_label = tf.expand_dims(image_pixel_label, 0)
            image_pixel_label = tf.expand_dims(image_pixel_label, 3)
            tf.summary.image('pixel_mask_0', image_pixel_label)
        
        link_pos_mask, link_neg_mask = get_pos_and_neg_masks(link_labels)
        # link_neg_scores = self.link_scores
        # n_link_pos = tf.reduce_sum(tf.cast(link_pos_mask, dtype = tf.float32))

        W = seg_selected_mask > 0.0 
        link_loss = []
        for i in range(8):
            with tf.name_scope('link_cls_loss_' + str(i)):
                # link_logits = self.link_cls[:,:,:,0:2]
                # link_labels = tf.cast(link_pos_mask[:,:,:,0], tf.int32)
                link_logits = self.link_cls[:,:,:,2 * i: 2 * i + 2]
                link_labels = tf.cast(link_pos_mask[:,:,:,i], tf.int32)
                # link_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=link_logits, labels=link_labels))
                link_ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=link_logits, labels=link_labels)

                # pos_weight = tf.cast(tf.logical_and(W, link_pos_mask[:,:,:,i]), tf.float32)
                # neg_weight = tf.cast(tf.logical_and(W, link_neg_mask[:,:,:,i]), tf.float32)
                pos_weight = tf.cast(link_pos_mask[:,:,:,i], tf.float32)
                neg_weight = tf.cast(link_neg_mask[:,:,:,i], tf.float32)
                pos_n = tf.reduce_sum(pos_weight)
                neg_n = tf.reduce_sum(neg_weight)

                def has_pos():
                    pos_scale = 1./pos_n
                    return link_ce_loss * pos_weight * pos_scale
                def no_pos():
                    return tf.zeros_like(link_ce_loss)

                def has_neg():
                    neg_scale = 1./neg_n
                    return link_ce_loss * neg_weight * neg_scale
                def no_neg():
                    return tf.zeros_like(link_ce_loss)

                link_pos_loss = tf.cond(tf.equal(pos_n, 0.0), no_pos, has_pos)
                link_neg_loss = tf.cond(tf.equal(neg_n, 0.0), no_neg, has_neg)
                link_cls_loss = tf.reduce_sum(link_pos_loss + link_neg_loss)

                # link_weighted_loss = tf.losses.compute_weighted_loss(link_cls_loss, seg_selected_mask)

                # link_weighted_loss = tf.Print(link_weighted_loss, [link_weighted_loss], 'link_weighted_loss' + str(i))


                # n_pos = tf.reduce_sum(pos_weight)
                # n_neg = tf.reduce_sum(neg_weight)

                # W_sum = tf.reduce_sum(tf.cast(W,tf.float32))

                # n_pos = tf.Print(n_pos, [n_pos,W_sum], 'n_pos,W_sum' + str(i))
                # n_neg = tf.Print(n_neg, [n_neg], 'n_neg')

                # ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=link_logits, labels=link_labels)
                
                # ce_loss = tf.Print(ce_loss, [ce_loss], 'ce_loss' + str(i))
                
                # tf.assert_equal(tf.shape(ce_loss), tf.shape(pos_weight))

                # def has_pos():
                    # return tf.reduce_sum(ce_loss * pos_weight) / n_pos
                
                # def has_neg():
                    # return tf.reduce_sum(ce_loss * neg_weight) / n_pos
                
                # def no():
                    # return tf.constant(0.0)

                # pos_loss = tf.cond(n_pos > 0, has_pos, no)
                # neg_loss = tf.cond(n_pos > 0, has_neg, no)

                # pos_loss = tf.Print(pos_loss, [pos_loss], 'pos_loss')
                # neg_loss = tf.Print(neg_loss, [neg_loss], 'neg_loss')

                # link_cls_loss = (pos_loss + neg_loss)
                link_loss.append(link_cls_loss)

                tf.summary.scalar('link_weighted_loss', link_cls_loss)

        link_total_loss = tf.add_n(link_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, link_total_loss)
        image_link_label = tf.cast(link_labels[0,:,:], tf.float32)
        image_link_label = tf.expand_dims(image_link_label, 0)
        image_link_label = tf.expand_dims(image_link_label, 3)
        tf.summary.image('link_mask_0', image_link_label)


        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            # tf.summary.scalar('link_cls_loss', link_cls_loss)


def tensor_shape(t):
    t.get_shape().assert_is_fully_defined()
    return t.get_shape().as_list()
        



