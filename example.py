import tensorflow as tf
import tensorflow.contrib.slim as slim

a = tf.constant([[[[1,2],[3,4],[5,6],[7,8]]]], dtype=tf.float32)
b = tf.constant([1,0,0,0])

d= tf.split(a, 2, 2)
d_shape = tf.shape(d)
a_shape = tf.shape(a)
b_shape = tf.shape(b)

a_regression = slim.softmax(a)
shape_regression = tf.shape(a_regression)

#calculate negative loss funcation

loss = - tf.reduce_mean(b * tf.cast(tf.log(a_regression), tf.int32))

with tf.Session() as sess:
    res_a_shape, res_b_shape= sess.run([a_shape, b_shape])
    print res_a_shape
    print res_b_shape
    print sess.run(d)
    print sess.run(d_shape)
    print sess.run(a_regression)
    print sess.run(shape_regression)
    print sess.run(loss)
