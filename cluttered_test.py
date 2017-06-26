# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import os

# %% Load data
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=11)


# %% Placeholders for 40x40 resolution
# x = tf.placeholder(tf.float32, [None, 1600])
x = tf.placeholder(tf.float32, [None, 8192])
y = tf.placeholder(tf.float32, [None, 10])

# %% Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
# x_tensor = tf.reshape(x, [-1, 40, 40, 1])
x_tensor = tf.reshape(x, [-1, 32, 256, 1])

tf.summary.image('image', x_tensor)

with tf.name_scope('SPATIAL_TRANS'):
    # %% We'll setup the two-layer localisation network to figure out the
    # %% parameters for an affine transformation of the input
    # %% Create variables for fully connected layer
    # W_fc_loc1 = weight_variable([1600, 20])
    W_fc_loc1 = weight_variable([8192, 20])
    b_fc_loc1 = bias_variable([20])

    W_fc_loc2 = weight_variable([20, 6])
    # Use identity transformation as starting point
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

    # %% Define the two layer localisation network
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
    # %% We can add dropout for regularizing and to reduce overfitting like so:
    keep_prob = tf.placeholder(tf.float32)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
    # %% Second layer
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

    # %% We'll create a spatial transformer module to identify discriminative
    # %% patches
    # out_size = (40, 40)
    out_size = (32, 256)
    h_trans = transformer(x_tensor, h_fc_loc2, out_size)

trans_tensor = tf.reshape(h_trans, [-1, 32, 256, 1])

# trans_tensor = tf.reshape(h_trans, [-1, 40, 40, 1])
tf.summary.image('image', trans_tensor)

# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]
with tf.name_scope('conv_1'):
    filter_size = 3
    n_filters_1 = 16
    W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

    # %% Bias is [output_channels]
    b_conv1 = bias_variable([n_filters_1])

    # %% Now we can build a graph which does the first layer of convolution:
    # we define our stride as batch x height x width x channels
    # instead of pooling, we use strides of 2 and more layers
    # with smaller filters.

    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(input=h_trans,
                     filter=W_conv1,
                     strides=[1, 2, 2, 1],
                     padding='SAME') +
        b_conv1)

with tf.name_scope('conv_2'):
    # %% And just like the first layer, add additional layers to create
    # a deep net
    n_filters_2 = 16
    W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
    b_conv2 = bias_variable([n_filters_2])
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(input=h_conv1,
                     filter=W_conv2,
                     strides=[1, 2, 2, 1],
                     padding='SAME') +
        b_conv2)

    # %% We'll now reshape so we can connect to a fully-connected layer:
    # h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 8192])

with tf.name_scope('fully_connect'):
    # %% Create a fully-connected layer:
    n_fc = 1024
    W_fc1 = weight_variable([8192, n_fc])
    b_fc1 = bias_variable([n_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('softmax'):
    # %% And finally our softmax layer:
    W_fc2 = weight_variable([n_fc, 10])
    b_fc2 = bias_variable([10])
    y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# %% Define loss/eval/training functions
with tf.name_scope('cross_enropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('optimizer'):
    opt = tf.train.AdamOptimizer()
    optimizer = opt.minimize(cross_entropy)

with tf.name_scope('gradients'):
    grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

with tf.name_scope('accuracy'):
    # %% Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    tf.summary.scalar('accuracy', accuracy)

# %% We now create a new session to actually perform the initialization the
# variables:
# sess = tf.Session()


    # with tf.name_scope('load_image'):
    # #读取图片
    #     file1 = tf.read_file('./ocr_data/1/3453.jpg')
    #     image = tf.image.decode_jpeg(file1, channels=1)
    #     image = tf.reshape(image, [1, 8192])
    #     writer= tf.summary.FileWriter("log", sess.graph)

    # with tf.name_scope('load_image_2'):
    # #读取图片
    #     file1 = tf.read_file('./ocr_data/2/2862.jpg')
    #     image_2 = tf.image.decode_jpeg(file1, channels=1)
    #     image_2 = tf.reshape(image_2, [1, 8192])
    #     writer= tf.summary.FileWriter("log", sess.graph)

    # with tf.name_scope('load_image_valid'):
    #     #读取图片
    #     file2 = tf.read_file('./ocr_data/2/2861.jpg')
    #     image_valid = tf.image.decode_jpeg(file1, channels=1)
    #     image_valid = tf.reshape(image_valid, [1, 8192])
    #     writer= tf.summary.FileWriter("log", sess.graph)

    def get_filenames(path):
        filenames = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if ".jpg" in f:
                    filenames.append(os.path.join(root, f))
        return filenames

    def read_img(filenames, num_epochs, shuffle=True):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        print(filename_queue)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        print('value', value)
        img = tf.image.decode_jpeg(value, channels=1)
        img = tf.reshape(img, [1, 8192])
        # img = tf.image.resize_images(img, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return img

    # %% We'll now train in minibatches and report accuracy, loss:
    iter_per_epoch = 10
    n_epochs = 500
    train_size = 10000

    # 将所有显示结果进行汇总
    merge = tf.summary.merge_all()

    #batch_xs = sess.run(image)
    image_list_1 = get_filenames('./ocr_data/1')
    print (image_list_1)

    img = read_img(image_list_1, 1, True)

    print(img)

    EPOCH = 10
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * 4

    img_batch = tf.train.batch([img], batch_size=EPOCH, num_threads=2,
                                  capacity=capacity)
    batch_ys = []
    for i in range(EPOCH):
        batch_ys.append(Y_train[i])

    # X_valid = sess.run(image_valid)

    with tf.Session() as sess:
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch_i in range(n_epochs):
        # try:
            batch_xs = sess.run([img_batch])
            loss,summaries = sess.run([cross_entropy, merge],
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(epoch_i) + ' Loss: ' + str(loss))

            writer.add_summary(summaries, epoch_i)
            sess.run(optimizer, feed_dict={
                x: batch_xs, y: batch_ys, keep_prob: 0.8})

            # print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
            #         feed_dict={ x: X_valid, y: Y_valid, keep_prob: 1.0 })))

        # except Exception, e:
        #     coord.request_stop(e)
        # finally:
        coord.request_stop()
        coord.join(threads)
    # theta = sess.run(h_fc_loc2, feed_dict={
    #        x: batch_xs, keep_prob: 1.0})
    # print(theta[0])
