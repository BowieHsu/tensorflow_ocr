import tensorflow as tf

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('test_data_path', './exhibition', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/res/','')

from nets import model
import pdb
import os
import cv2
import time
import numpy as np

import pdb

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

def pixel_detect(score_map, geo_map, score_map_thresh=0.8, link_thresh=0.8):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    res_map = np.zeros((score_map.shape[0] ,score_map.shape[1] ))
    ori_res_map = np.zeros((score_map.shape[0] * 4, score_map.shape[1] * 4))
    xy_text = np.argwhere(score_map > score_map_thresh)

    for p in xy_text:
        res_map[p[0], p[1]] = 1

    res = res_map

    for i in range(8):
        geo_map_split = geo_map[:,:,i * 2 - 1]
        link_text = np.argwhere(geo_map_split > link_thresh)
        # pdb.set_trace()
        # for p in link_text:
            # res[p[0],p[1]] = 1

    xy_text = np.argwhere(res == 1)

    for p in xy_text:
        ori_res_map[p[0] * 4 - 2: p[0] * 4 + 2, p[1] * 4 - 2 : p[1] * 4 + 2] = 1

    ori_res_map = np.array(ori_res_map, dtype=np.uint8)
   
    return ori_res_map

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

def resize_image(im, max_side_len=1000):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    print 'resize'
    print resize_h
    print resize_w
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        timer = {'net':0}

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                print ratio_h,ratio_w
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start_time

                print 'net time:' + str(timer['net'] * 1000) + 'ms'

                score_map_res = pixel_detect(score_map=score, geo_map=geometry)

                # pdb.set_trace()

                im2, contours , hierarchy = cv2.findContours(score_map_res,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # pdb.set_trace()

                for i in range(len(contours)):
                    np_contours = np.array(np.reshape(contours[i],[-1,2]), dtype=np.float32)

                    rectangle = cv2.minAreaRect(np_contours)

                    box = np.int0(cv2.boxPoints(rectangle))
                    cv2.drawContours(im_resized, [box], -1,(0,0,255), 3)

                # cv2.polylines(im_resized, points, -1, (0,0,255), 3)

                img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))

                cv2.imwrite(img_path, im_resized)

if __name__ == '__main__':
    tf.app.run()
