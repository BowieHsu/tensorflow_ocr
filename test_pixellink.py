import tensorflow as tf
import cv2

import config
from nets import pixellink
from datasets import ssd_vgg_preprocessing
from tf_extended import pixellink_fn
import os
import pdb
import numpy as np

tf.app.flags.DEFINE_float('pixel_conf_threshold', 0.8, 'the threshold on the confidence of segment')
tf.app.flags.DEFINE_float('link_conf_threshold', 0.8, 'the threshold on the confidence of link')
tf.app.flags.DEFINE_string('test_data_path', './icdar_test/', '')
tf.app.flags.DEFINE_string('checkpoint_path', './ohem_logs/', '')
tf.app.flags.DEFINE_string('output_dir', './tmp/','')
tf.app.flags.DEFINE_integer('image_width', 1280, '')
tf.app.flags.DEFINE_integer('image_height', 768, '')
tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, '')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    image_shape = (FLAGS.image_height, FLAGS.image_width)

    config.init_config(image_shape, batch_size = 1, pixel_conf_threshold = FLAGS.pixel_conf_threshold, link_conf_threshold = FLAGS.link_conf_threshold)

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def eval():
    with tf.name_scope('test'):
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):
            tf_image = tf.placeholder(dtype = tf.int32, shape = [None, None, 3])
            image_shape = tf.placeholder(dtype = tf.int32, shape = [3,])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(tf_image, None, None, None, None, out_shape = config.image_shape, data_format = 'NHWC', is_training = False)
            b_image = tf.expand_dims(processed_image, axis = 0)
            net = pixellink.PixelLinkNet(inputs = b_image, data_format = config.data_format)

            ori_pixel_score = tf.nn.softmax(net.pixel_cls)[:,:,:,1],
            pixel_score = tf.expand_dims(tf.nn.softmax(net.pixel_cls)[:,:,:,1],3)
            link_score = []
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,0:2]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,2:4]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,4:6]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,6:8]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,8:10]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,10:12]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,12:14]))
            link_score.append(tf.nn.softmax(net.link_cls[:,:,:,14:16]))
            link_score_res = tf.stack(link_score)
            # pdb.set_trace()
            score_res = pixellink_fn.tf_pixel_detect(pixel_score, link_score_res, FLAGS.pixel_conf_threshold, FLAGS.link_conf_threshold)

    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    saver = tf.train.Saver()
    #get checkpoint file based on path
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
    checkpoint_path = checkpoint.model_checkpoint_path
    tf.logging.info('testing', checkpoint)


    with tf.Session(config = sess_config) as sess:
        saver.restore(sess, checkpoint_path)
        image_names = get_images()
        #test image 
        for iter, image_name in enumerate(image_names):
            print image_name
            boxes = []
            im = cv2.imread(image_name)[:,:, ::-1]
            tf_score_res = sess.run([score_res], feed_dict = {tf_image:im, image_shape:im.shape})
            b_pixel_score = sess.run([ori_pixel_score], feed_dict = {tf_image:im, image_shape:im.shape})
            b_link_score = sess.run([link_score_res], feed_dict = {tf_image:im, image_shape:im.shape})
            # pdb.set_trace()
            im_ori = cv2.imread(image_name)
            link_score_set = []
            for i in range(8):
                b_score = np.array(b_link_score[0][i], dtype = np.float32)
                b_score = b_score[0,:,:,1] * 255
                link_score = cv2.resize(b_score, (1280, 720), interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(link_score, cv2.COLOR_GRAY2BGR)
                gray = gray * 0.5 + im_ori * 0.5
                # link_score = link_score > 235
                # link_score = np.array(link_score, dtype = np.uint8)
                # pdb.set_trace()
                link_score_set.append(link_score)
                cv2.imwrite('./link_score_' + str(i) + '.jpg', gray)
            # b_process_image = sess.run([processed_image], feed_dict = {tf_image:im, image_shape:im.shape})
            pixel_score = np.array(b_pixel_score[0][0][0] , dtype = np.float32)
            pixel_score = cv2.resize(pixel_score, (1280, 720), interpolation=cv2.INTER_CUBIC)
            pixel_score = pixel_score * 255
            
            # gray = cv2.cvtColor(pixel_score, cv2.COLOR_GRAY2BGR)
            # gray = gray * 0.5 + im_ori * 0.5
            pixel_seg = pixel_score > int(255 * 0.8) 
            cv2.imwrite('./pixel_score.jpg', np.array(pixel_seg, dtype=np.uint8)* 255)

            #try to link pixel using link scores
            test_score = np.zeros((720, 1280))
            group_idx = np.zeros(720 * 1280)
            graph = {}
            thresh = 255 * 0.9
            pixel_thresh = 255 * 0.8
            for x in range(1, 1279):
                for y in range(1, 719):
                    neighbor = []
                    # if pixel_score[y, x] > pixel_thresh:
                    if pixel_seg[y][x]:
                        if link_score_set[0][y][x] > thresh and pixel_seg[y][x-1]: #left
                            neighbor.append(y * 1280 + x - 1)

                        if link_score_set[1][y][x] > thresh and pixel_seg[y + 1][x - 1]: #left down
                            neighbor.append((y + 1) * 1280 +  x - 1)

                        if link_score_set[2][y][x] > thresh and pixel_seg[y - 1][x - 1]: #left up
                            neighbor.append((y - 1) * 1280 + x - 1)

                        if link_score_set[3][y][x] > thresh and pixel_seg[y][x + 1]: #right
                            neighbor.append(y * 1280 + x + 1)

                        if link_score_set[4][y][x] > thresh and pixel_seg[y + 1][x + 1]: #right down
                            neighbor.append((y + 1) * 1280 + x + 1)

                        if link_score_set[5][y][x] > thresh and pixel_seg[y - 1][x + 1]: #right up
                            neighbor.append((y - 1) * 1280 + x + 1)

                        if link_score_set[6][y][x] > thresh and pixel_seg[y - 1][x]: #up
                            neighbor.append((y - 1) * 1280 + x)

                        if link_score_set[7][y][x] > thresh and pixel_seg[y + 1][x]: #down
                            neighbor.append((y + 1) * 1280 + x)

                        test_score[y, x] = 255
                        # print neighbor
                        graph[y * 1280 + x] = neighbor

            gid = 1
            def dfs(graph, v):
                # print 'group_idx',np.sum(group_idx)
                if group_idx[v] != 0.0:
                    return []
                S = []
                S.append(v)
                label = []
                while S:
                    v = S.pop()
                    if v not in label:
                        label.append(v)
                        if graph.has_key(v):
                            for edge in graph[v]:
                                if group_idx[edge] == 0.0:
                                    S.append(edge)
                return label


            for i in graph.keys():
                print 'gid',gid
                index_list = dfs(graph, i)
                if len(index_list) > 200:
                    print index_list
                    for index in index_list:
                        group_idx[index] = gid
                    gid += 1

            group_show = np.true_divide(group_idx, gid)

            # cv2.imwrite('./test_seg.jpg', test_score)
            group_show = np.reshape(group_show, (720, 1280))
            group = np.array(group_show, dtype=np.float32) * 255
            # cv2.imwrite('./group_idx.jpg', group)

            b_score_res = np.array(tf_score_res, dtype = np.uint8)[0,:,:]
            res = cv2.resize(b_score_res, (1280, 720), interpolation=cv2.INTER_CUBIC)
            pixel_res = res * 255
            # cv2.imwrite('./tf_img.jpg', pixel_res)

            # im2, contours , hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # for i in range(len(contours)):
                # np_contours = np.array(np.reshape(contours[i],[-1,2]), dtype=np.float32)
                # rectangle = cv2.minAreaRect(np_contours)

                # box = np.int0(cv2.boxPoints(rectangle))
                # # if 1:
                # if rectangle[1][0] > 10 and rectangle[1][1] > 10:
                    # cv2.drawContours(im_ori, [box], -1,(0,255,0), 1)
                    # boxes.append(box)
            group_idx = np.reshape(group_idx, (720, 1280))

            for i in range(1,gid):
                xy_in_poly = np.argwhere(group_idx == i)
                show_xy = xy_in_poly.copy()
                show_xy[:,0] = xy_in_poly[:,1]
                show_xy[:,1] = xy_in_poly[:,0]
                # pdb.set_trace()
                rectangle = cv2.minAreaRect(show_xy)
                box = np.int0(cv2.boxPoints(rectangle))
                cv2.drawContours(im_ori, [box], -1,(0,255,0), 1)
                boxes.append(box)

            img_path = os.path.join(FLAGS.output_dir, os.path.basename(image_name))
            cv2.imwrite(img_path, im_ori)

            if boxes is not None:
               res_file = os.path.join(
                   FLAGS.output_dir,
                   'res_{}.txt'.format(
                       os.path.basename(image_name).split('.')[0]))
               
               with open(res_file,'w') as f:
                   for box in boxes:
                       f.write('{},{},{},{},{},{},{},{}\r\n'.format( box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))

def main(_):
    config_initialization()
    eval()

if __name__ == '__main__':
    tf.app.run()
