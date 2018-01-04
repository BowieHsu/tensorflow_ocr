import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory 
from datasets import ssd_vgg_preprocessing
from nets import pixellink
from tf_extended import pixellink_fn
import config
import pdb
import os
from tensorflow.python import debug as tf_debug

slim = tf.contrib.slim

# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_bool('train_with_ignored', False, 
                           'whether to use ignored bbox (in ic15) in training.')
tf.app.flags.DEFINE_float('pixel_cls_loss_weight', 1.0, 'the loss weight of segment localization')
tf.app.flags.DEFINE_float('link_cls_loss_weight', 1.0, 'the loss weight of linkage classification loss')

tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_gpus', 2, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 60000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'log frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', 'pixellink_layers', 'checkpoint_exclude_scopes')

# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('lr_policy', 'staircase', 'learning rate.')
tf.app.flags.DEFINE_string('lr_breakpoints', '20000,40000,60000', 'learning rate.')
tf.app.flags.DEFINE_string('lr_decays', '0.1,0.01,0.001', 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', False, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 32,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 32,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'seglink_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer('train_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    #util.init_logger()
    print(FLAGS.batch_size)

    config.init_config(image_shape, batch_size = FLAGS.batch_size, 
                       weight_decay = FLAGS.weight_decay, 
                       num_gpus = FLAGS.num_gpus,
                       train_with_ignored = FLAGS.train_with_ignored)

    batch_size_per_gpu = int(FLAGS.batch_size/FLAGS.num_gpus)

    tf.summary.scalar('batch_size', FLAGS.batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    return dataset

def create_dataset_batch_queue(dataset):
    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        num_readers=FLAGS.num_readers,
                        common_queue_capacity = 50 * config.batch_size,
                        common_queue_min = 30 * config.batch_size,
                        shuffle=True)

            [image, ignored, bboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
                                                                    'image',
                                                                    'object/ignored',
                                                                    'object/bbox',
                                                                    'object/oriented_bbox/x1',
                                                                    'object/oriented_bbox/x2',
                                                                    'object/oriented_bbox/x3',
                                                                    'object/oriented_bbox/x4',
                                                                    'object/oriented_bbox/y1',
                                                                    'object/oriented_bbox/y2',
                                                                    'object/oriented_bbox/y3',
                                                                    'object/oriented_bbox/y4'])

            gxs = tf.transpose(tf.stack([x1,x2,x3,x4]))
            gys = tf.transpose(tf.stack([y1,y2,y3,y4]))
            image = tf.identity(image, 'input_image')

            image, ignored, bboxes, gxs, gys = ssd_vgg_preprocessing.preprocess_image(image, ignored, bboxes, gxs, gys, 
                                                            out_shape = (512,512),
                                                            data_format = config.data_format,
                                                            is_training = True)
            image = tf.identity(image, 'processed_image')

            pdb.set_trace()
            #place pixel link ground truth
            pixel_labels, link_labels = pixellink_fn.tf_pixellink_get_rbox((512, 512), gxs, gys, ignored)

            # pixel_labels_image = tf.expand_dims(pixel_labels, 0)
            # pixel_labels_image = tf.expand_dims(pixel_labels_image, 3)
            # tf.summary.image('pixel_labels', pixel_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,0], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_0', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,1], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_1', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,2], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_2', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,3], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_3', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,4], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_4', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,5], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_5', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,6], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_6', link_labels_image)

            # link_labels_image = tf.expand_dims(link_labels[:,:,7], 0)
            # link_labels_image = tf.expand_dims(link_labels_image, 3)
            # tf.summary.image('link_labels_7', link_labels_image)

            b_image, b_pixel_labels, b_link_labels = tf.train.batch([image, pixel_labels, link_labels], batch_size = FLAGS.batch_size/FLAGS.num_gpus, num_threads = FLAGS.num_preprocessing_threads, capacity = 50)

            batch_queue = slim.prefetch_queue.prefetch_queue([b_image, b_pixel_labels, b_link_labels], capacity = 50)

    return batch_queue

def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            grads.append(g)
        grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        averaged_grads.append((grad, v))
        
        tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
        tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
        tf.summary.scalar("variables_and_gradients_" + grad.op.name+'_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
        tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean', tf.reduce_mean(var))
    return averaged_grads

# def _setup_train_net_multigpu(self):
 # 99     with tf.device('/cpu:0'):
 # 100       # learning rate decay
 # 101       with tf.name_scope('lr_decay'):
 # 102         if FLAGS.lr_policy == 'staircase':
 # 103           # decayed learning rate
 # 104           lr_breakpoints = [int(o) for o in FLAGS.lr_breakpoints.split(',')]
 # 105           lr_decays = [float(o) for o in FLAGS.lr_decays.split(',')]
 # 106           assert(len(lr_breakpoints) == len(lr_decays))
 # 107           pred_fn_pairs = []
 # 108           for lr_decay, lr_breakpoint in zip(lr_decays, lr_breakpoints):
 # 109             fn = (lambda o: lambda: tf.constant(o, tf.float32))(lr_decay)
 # 110             pred_fn_pairs.append((tf.less(self.global_step, lr_breakpoint), fn))
 # 111           lr_decay = tf.case(pred_fn_pairs, default=(lambda: tf.constant(1.0)))
 # 112         else:
 # 113           logging.error('Unkonw lr_policy: {}'.format(FLAGS.lr_policy))
 # 114           sys.exit(1)
 # 115
 # 116         self.current_lr = lr_decay * FLAGS.base_lr
 # 117         tf.summary.scalar('lr', self.current_lr, collections=['brief'])
 

def create_clones(batch_queue):
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        with tf.name_scope('lr_decay'):
            if FLAGS.lr_policy == 'staircase':
                # decayed learning rate
                lr_breakpoints = [int(o) for o in FLAGS.lr_breakpoints.split(',')]
                lr_decays = [float(o) for o in FLAGS.lr_decays.split(',')]
                assert(len(lr_breakpoints) == len(lr_decays))
                pred_fn_pairs = []
                for lr_decay, lr_breakpoint in zip(lr_decays, lr_breakpoints):
                  fn = (lambda o: lambda: tf.constant(o, tf.float32))(lr_decay)
                  pred_fn_pairs.append((tf.less(global_step, lr_breakpoint), fn))
                lr_decay = tf.case(pred_fn_pairs, default=(lambda: tf.constant(1.0)))
            else:
                logging.error('Unkonw lr_policy: {}'.format(FLAGS.lr_policy))
                sys.exit(1)
 
            # pdb.set_trace()
            current_lr = lr_decay * FLAGS.learning_rate
            tf.summary.scalar('lr', current_lr)

        # learning_rate = tf.constant(FLAGS.learning_rate, name = 'learning_rate')
        # learning_rate = tf.constant(current_lr, name = 'learning_rate')
        # tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(current_lr, momentum=FLAGS.momentum, name='Momentum')

    pixellink_loss = 0; # for summary only
    gradients = []

    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0
        with tf.variable_scope(tf.get_variable_scope(), reuse = True): # the variable has been created in config.init_config
            with  tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    b_image, b_pixel_label, b_link_label = batch_queue.dequeue()
                    net = pixellink.PixelLinkNet(inputs = b_image, data_format = config.data_format)

                    net.build_loss(pixel_labels = b_pixel_label,
                                   link_labels = b_link_label,
                                   do_summary = do_summary)

                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                        
                    pdb.set_trace()
                    assert len(losses) == 2
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    pixellink_loss = pixellink_loss + total_clone_loss

                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss

                    clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)
    tf.summary.scalar('pixellink_loss', pixellink_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    averaged_gradients = sum_gradients(gradients)
    update_op = optimizer.apply_gradients(averaged_gradients, global_step = global_step)

    train_ops = [update_op]

    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training,\
                        with decay = %f'%(FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([update_op]):
            train_ops.append(tf.group(ema_op))

    train_op = control_flow_ops.with_dependencies(train_ops, pixellink_loss, name='train_op')
    return train_op

def get_latest_ckpt(path):
    if os.path.isdir(path):
        ckpt = tf.train.get_checkpoint_state(path)
        ckpt_path = ckpt.model_checkpoint_path
    else:
        ckpt_path = path
    return ckpt_path

def get_init_fn(checkpoint_path, train_dir, ignore_missing_vars = False, 
                checkpoint_exclude_scopes = None, model_name = None, checkpoint_model_scope = None):
    """
    code from github/SSD-tensorflow/tf_utils.py
    Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    checkpoint_path: the checkpoint to be restored
    train_dir: the directory where checkpoints are stored during training.
    ignore_missing_vars: if False and there are variables in the model but not in the checkpoint, an error will be raised.
    checkpoint_model_scope and model_name: if the root scope of checkpoints and the model in session is different, 
            (but the sub-scopes are all the same), specify them clearly 
    checkpoint_exclude_scopes: variables to be excluded when restoring from checkpoint_path.
    Returns:
      An init function run by the supervisor.
    """
    # if util.str.is_none_or_empty(checkpoint_path):
    if (checkpoint_path is None) or (len(checkpoint_path) == 0):
        return None
        # return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if checkpoint_model_scope is not None:
        variables_to_restore = {var.op.name.replace(model_name, checkpoint_model_scope): var for var in variables_to_restore}

    checkpoint_path = get_latest_ckpt(checkpoint_path)
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)


def train(train_op):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)

    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True

    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    init_fn = get_init_fn(checkpoint_path = FLAGS.checkpoint_path, train_dir = FLAGS.train_dir, ignore_missing_vars = FLAGS.ignore_missing_vars, checkpoint_exclude_scopes = FLAGS.checkpoint_exclude_scopes)
    # init_fn = slim.assign_from_checkpoint_fn(FLAGS.checkpoint_path, variables_to_restore, ignore_missing_vars=FLAGS.ignore_missing_vars)
    
    saver = tf.train.Saver(max_to_keep = 500, write_version = 2)

    # init = tf.global_variables_initializer()

    # with tf.Session(config= sess_config) as sess:     
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.run(init)
        # init_fn(sess)
        # for step in range(10):
            # b_op = sess.run(train_op)
    slim.learning.train(train_op, logdir=FLAGS.train_dir, init_fn = init_fn, summary_op = summary_op, number_of_steps = FLAGS.max_number_of_steps, log_every_n_steps = FLAGS.log_every_n_steps, save_summaries_secs = 5, saver = saver, save_interval_secs = 300, session_config = sess_config)
    # session_wrapper=tf_debug.LocalCLIDebugWrapperSession

def main(_):
    dataset = config_initialization()
    batch_queue = create_dataset_batch_queue(dataset)
    train_op = create_clones(batch_queue)
    train(train_op)


if __name__ == '__main__':
    tf.app.run()
