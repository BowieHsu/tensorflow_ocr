import cv2
import numpy as np
import tensorflow as tf

import config
import pdb


def valid_link(x, y, score_map, val, w, h, direction):
    if x == w - 1 or y == h - 1 or x == 0 or y == 0:
        return 1.0
    if direction == 'up':
        new_x = x
        new_y = y - 1
        # point_dir = np.array([point[0], point[1] - 1])
    elif direction == 'down':
        new_x = x
        new_y = y + 1
        # point_dir = np.array([point[0], point[1] + 1])
    elif direction == 'left':
        new_x = x - 1
        new_y = y 
        # point_dir = np.array([point[0] - 1, point[1]])
    elif direction == 'right':
        new_x = x + 1
        new_y = y 
        # point_dir = np.array([point[0] + 1, point[1]])
    elif direction == 'left_up':
        new_x = x - 1
        new_y = y - 1
        # point_dir = np.array([point[0] - 1, point[1] - 1])
    elif direction == 'left_down':
        new_x = x - 1
        new_y = y + 1
        # point_dir = np.array([point[0] - 1, point[1] + 1])
    elif direction == 'right_up':
        new_x = x + 1
        new_y = y - 1
        # point_dir = np.array([point[0] + 1, point[1] - 1])
    elif direction == 'right_down':
        new_x = x + 1
        new_y = y + 1
        # point_dir = np.array([point[0] + 1, point[1] + 1])
    if score_map[new_y, new_x] == val:
        return 1.0
    else:
        return 0.0

def points_to_contour(points):
    contours = [[list(p)] for p in points]
    return np.asarray(contours, dtype = np.int32)

def generate_rbox(h, w,  xs, ys, ignored):
    assert len(xs) == len(ignored), 'the length of xs and ignored must be the same,\
        but got %s and %s'%(len(xs), len(ignored))
    h = h/4
    w = w/4
    xs = xs
    ys = ys
    score_map = np.zeros((h, w), dtype=np.float32)
    link_map = np.zeros((h, w, 8), dtype=np.float32)
    poly_mask = np.zeros((h,w), dtype=np.uint8)

    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)


    num_rects = xs.shape[0]
    rect_area = []
    poly = []

    # for idx in range(num_rects):
    # if valid_bbox_idxes == []:
        # return score_map, link_map

    for idx in range(num_rects):
        # if idx in valid_bbox_idxes:
            points = zip(xs[idx, :] * w, ys[idx, :] * h)
            cnt = points_to_contour(points)
            rect = cv2.minAreaRect(cnt)
            poly.append(np.int0(cv2.boxPoints(rect)))
            area = rect[1][0] * rect[1][1]
            rect_area.append(area)
            if rect[1][0] < 10 or rect[1][1] < 10:
                ignored[idx] = 0

    # B = np.sum(rect_area)/num_rects * 1.0
    # rect_area = np.true_divide(B, rect_area)

    valid_bbox_idxes = np.where(ignored == 0)[0]

    for poly_idx in range(num_rects):
        draw_poly = poly[poly_idx].astype(np.int32)[np.newaxis, :,:]
        # cv2.fillPoly(score_map, draw_poly, rect_area[poly_idx])
        if(poly_idx not in valid_bbox_idxes) and (valid_bbox_idxes != []):
            cv2.fillPoly(score_map, draw_poly, -1.0)
        else:
            cv2.fillPoly(score_map, draw_poly, 1.0)
            cv2.fillPoly(poly_mask, draw_poly, poly_idx + 1)
            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

            for y, x in xy_in_poly:
                # point = np.array([x, y], dtype=np.int32)
                # left
                link_map[y, x, 0] = valid_link(x, y, poly_mask, poly_idx + 1, w, h,'left')
                # left_down
                link_map[y, x, 1] = valid_link(x, y, poly_mask, poly_idx + 1, w, h, 'left_down')
                # left_up
                link_map[y, x, 2] = valid_link(x, y, poly_mask, poly_idx + 1, w, h, 'left_up')
                # right
                link_map[y, x, 3] = valid_link(x, y, poly_mask, poly_idx + 1, w, h,'right')
                # right_down
                link_map[y, x, 4] = valid_link(x, y, poly_mask, poly_idx + 1, w, h,'right_down')
                # right_up
                link_map[y, x, 5] = valid_link(x, y, poly_mask, poly_idx + 1, w, h, 'right_up')
                # up
                link_map[y, x, 6] = valid_link(x, y, poly_mask, poly_idx + 1, w, h, 'up')
                # down
                link_map[y, x, 7] = valid_link(x, y, poly_mask, poly_idx + 1, w, h, 'down')

    return score_map, link_map

def tf_pixellink_get_rbox(img_size, xs, ys, ignored):
    h, w = img_size
    pixel_map, link_map = tf.py_func(generate_rbox, [h, w, xs, ys, ignored], [tf.float32, tf.float32])
    pixel_map.set_shape([h/4, w/4])
    link_map.set_shape([h/4 ,w/4, 8])
    return pixel_map, link_map

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
        geo_map_solo = geo_map[:, 0, :, :, ]

    # filter the score map
    res_map = np.zeros((score_map.shape[0] ,score_map.shape[1] ))
    xy_text = np.argwhere(score_map > score_map_thresh)

    for p in xy_text:
        res_map[p[0], p[1]] = 1

    # res = res_map

    for i in range(8):
        # print i
        geo_map_split = geo_map_solo[i,:,:,1]
        link_text = np.argwhere(geo_map_split < link_thresh)
        link_num = len(link_text)
        for idx in range(link_num):
            res_map[link_text[idx][0], link_text[idx][1]] = 0
        # print np.sum(res_map > 0)
   
    # res = np.zeros((score_map.shape[0] ,score_map.shape[1] ))
    return np.array(res_map, dtype=np.uint8) 

def tf_pixel_detect(score_map, geo_map, score_map_thresh, link_thresh):
    res_map = tf.py_func(pixel_detect, [score_map, geo_map, score_map_thresh, link_thresh], tf.uint8) 
    return res_map
