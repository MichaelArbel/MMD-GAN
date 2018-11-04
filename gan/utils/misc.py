"""
Some codes from https://github.com/Newmu/dcgan_code

Released under the MIT license.
"""
from __future__ import division
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import tensorflow as tf
from six.moves import xrange
import os
import math
import operator
pp = pprint.PrettyPrinter()

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt


def inverse_transform(images):
    return (images+1.)/2.


def save_images(images, size, image_path):
    merged = merge(inverse_transform(images), size)
    return scipy.misc.imsave(image_path, merged)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        time0 = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        save_images(samples, [8, 8], './samples/test_%s.png' % time0)
    elif option == 1:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
        num_col = 8
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        samples = samples[:num_col**2, :, :, :]
        plot_dir = os.path.join(dcgan.sample_dir, 'figures')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        save_images(samples, [num_col, num_col], os.path.join(plot_dir, 'img-'+dcgan.config.ckpt_name+'.png'))
    elif option == 2:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 3:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1./config.batch_size)

        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

        image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
        make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))
    elif option == 5:
        plot_dir = os.path.join(dcgan.sample_dir, 'filters', dcgan.config.ckpt_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        #conv_weights = sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator/d_h6_lin/Matrix:0')])
        conv_weights_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        conv_weights_op = [w for w in conv_weights_op if 'w:'in w.name]
        conv_weights = sess.run([conv_weights_op])
        for i, c in enumerate(conv_weights[0]):
            name = 'ckpt_' + dcgan.config.ckpt_name + '_conv{}'.format(i)
            np.save(os.path.join(plot_dir, name), c)
            #plot_conv_weights(c, name,plot_dir)

    #new_image_set = [
    #    merge(np.array([images[idx] for images in image_set]), [10, 10])
    #    for idx in range(64) + range(63, -1, -1)]
    #make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def unpickle(file):
    fo = open(file, 'rb')
    try:  # Python 3 compatibility
        import _pickle as pickle
        dict = pickle.load(fo, encoding='latin1')
    except:  # Python 2
        import cPickle as pickle
        dict = pickle.load(fo)
    fo.close()
    return dict


def center_and_scale(im, size=64):
    size = int(size)
    arr = np.array(im)
    scale = min(im.size)/float(size)
    new_size = np.array(im.size)/scale
    im.thumbnail(new_size)
    arr = np.array(im)
    assert min(arr.shape[:2]) == size, "shape error: " + repr(arr.shape) + ", lower dim should be " + repr(size)
#    l0 = int((arr.shape[0] - size)//2)
#    l1 = int((arr.shape[1] - size)//2)
    l0 = np.random.choice(np.arange(arr.shape[0] - size + 1), 1)[0]
    l1 = np.random.choice(np.arange(arr.shape[1] - size + 1), 1)[0]
    arr = arr[l0:l0 + size, l1: l1 + size, :]
    sh = (size, size, 3)
    assert arr.shape == sh, "shape error: " + repr(arr.shape) + ", should be " + repr(sh)
    return np.asarray(arr/255., dtype=np.float32)


def center_and_scale_new(im, size=64, assumed_input_size=256, channels=3):
    if assumed_input_size is not None:
        ratio = int(assumed_input_size/size)
        decoded = tf.image.decode_jpeg(im, channels=channels, ratio=ratio)
        cropped = tf.random_crop(decoded, size=[size, size, 3])
        return tf.to_float(cropped)/255.
    size = int(size)
    decoded = tf.image.decode_jpeg(im, channels=channels)
    s = tf.reduce_min(tf.shape(decoded)[:2])
    cropped = tf.random_crop(decoded, size=[s, s, 3])
    scaled = tf.image.resize_images(cropped, [size, size])
    return tf.to_float(scaled)/255.


def read_and_scale(file, size=64):
    from PIL import Image
    im = Image.open(file)
    return center_and_scale(im, size=size)


def variable_summary(var, name, with_singular_values=False):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#    with tf.get_variable_scope():
    if var is None:
        print("Variable Summary: None value for variable '%s'" % name)
        return

    if with_singular_values:
       shape_var =  var.get_shape().as_list()
       col = np.prod(np.array(shape_var[:-1]))
       new_var = tf.reshape(tf.transpose(var, [3,0,1,2] ), [shape_var[-1],col])
       svd = tf.svd(new_var, compute_uv=False)
       svd = svd/svd[0]
       shape_s = svd.get_shape().as_list()
       condition_num = svd[0]/svd[shape_s[0]-1]
       tf.summary.scalar(name + '_condition_number', condition_num)
       tf.summary.tensor_summary(name + '_singular_val', svd)

    var = tf.clip_by_value(var, -1000., 1000.)
    mean = tf.reduce_mean(var)
    with tf.name_scope('absdev'):
        stddev = tf.reduce_mean(tf.abs(var - mean))


    tf.summary.scalar(name + '_absdev', stddev)
    tf.summary.histogram(name + '_histogram', var)


def variable_summaries(variable_dict,with_singular_values=False):
    for name, var in variable_dict.items():
        variable_summary(var, name,with_singular_values=with_singular_values)


def conv_sizes(size, layers, stride=2):
    s = [int(size)]
    for l in range(layers):
        s.append(int(np.ceil(float(s[-1])/float(stride))))
    return tuple(s)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width,
                                    resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/255.


def tf_read_jpeg(files, base_size=160, target_size=64, batch_size=128,
                 capacity=4000, num_threads=4, random_crop=9):
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.WholeFileReader()
    _, raw = reader.read(filename_queue)
    decoded = tf.image.decode_jpeg(raw, channels=3)  # HWC
    bs = base_size + 2 * random_crop
    cropped = tf.image.resize_image_with_crop_or_pad(decoded, bs, bs)
    if random_crop > 0:
        cropped = tf.image.random_flip_left_right(cropped)
        cropped = tf.random_crop(cropped, [base_size, base_size, 3])
    ims = tf.train.shuffle_batch(
        [cropped],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=capacity//4,
        num_threads=4,
        enqueue_many=False
    )

    resized = tf.image.resize_bilinear(ims, (target_size, target_size))
    images = tf.cast(resized, tf.float32)/255.
    return images


def PIL_read_jpeg(files, base_size=160, target_size=64, batch_size=128,
                  capacity=4000, num_threads=4):
    from PIL import Image

    def read_single(f):
        img = Image.open(f)
        w, h = img.size
        assert w >= base_size, 'wrong width'
        assert h >= base_size, 'wrong height'
        l, r = (w - base_size)//2, (h - base_size)//2
        img.crop((l, r, l + base_size, r + base_size))
        img.resize((target_size, target_size), Image.ANTIALIAS)
        return np.asarray(img, tf.float32)/255.

    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    single_file = filename_queue.dequeue()
    single_sample = tf.py_func(read_single, [single_file], tf.float32)
    single_sample.set_shape([target_size, target_size, 3])

    images = tf.train.shuffle_batch(
        [single_sample],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=capacity//4,
        num_threads=4,
        enqueue_many=False
    )

    return images


def plot_conv_weights(weights, name, plot_dir, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(plot_dir, 'conv_weights', name)
    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    w_min = -0.2
    w_max = 0.2
    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in xrange(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Warning: {}'.format(e))


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)


def viz_filters(sess, gan):
    plot_dir = os.path.join(gan.sample_dir, 'filters')
    conv_weights = sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator/d_h0_conv/w:0')])
    for i, c in enumerate(conv_weights[0]):
        plot_conv_weights(c, 'conv{}'.format(i), plot_dir)


class GpuParamServerDeviceSetter(object):
    """Used with tf.device() to place variables on the least loaded GPU.
    A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
    'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
    placed on the least loaded gpu. All other Ops, which will be the computation
    Ops, will be placed on the worker_device.
    """

    def __init__(self, worker_device, ps_devices):
        """Initializer for GpuParamServerDeviceSetter.
        Args:
            worker_device: the device to use for computation Ops.
            ps_devices: a list of devices to use for Variable Ops. Each variable is
            assigned to the least loaded device.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
            return self.worker_device

        # Gets the least loaded ps_device
        device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


def _create_device_setter(is_cpu_ps, worker, num_gpus, ps_device=''):
    """Create device setter object."""
    if is_cpu_ps:
        # tf.train.replica_device_setter supports placing variables on the CPU, all
        # on one GPU, or on ps_servers defined in a cluster_spec.
        return tf.train.replica_device_setter(
            worker_device=worker, ps_device=ps_device, ps_tasks=1)
    else:
        gpus = ['/gpu:%d' % i for i in range(num_gpus)]
        return GpuParamServerDeviceSetter(worker, gpus)
