import tensorflow as tf

a = tf.constant([[1,2],[3,4],[5,6]], dtype=tf.float32)
b = tf.constant([1,0,0])
a_shape = tf.shape(a)
b_shape = tf.shape(b)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=b)

with tf.Session() as sess:
    res_loss, res_a_shape, res_b_shape = sess.run([loss, a_shape, b_shape])
    print res_loss
    print res_a_shape
    print res_b_shape
