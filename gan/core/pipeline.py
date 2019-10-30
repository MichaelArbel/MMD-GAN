#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:11:46 2018

@author: mikolajbinkowski
"""
import os
import time
import lmdb
import io
import numpy as np
import tensorflow as tf
from utils import misc
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.python.ops.data_flow_ops import RecordInput, StagingArea


random = True
if random:
    Record_seed = np.random.randint(1,500)
else:
    Record_seed = 301

class Pipeline(object):
    def __init__(self, output_size, c_dim, batch_size, data_dir, format='NCHW', with_labels=False, **kwargs):
        self.output_size = output_size
        self.c_dim = c_dim
        self.batch_size = batch_size
        self.read_batch = max(4000, batch_size * 10)
        self.read_count = 0
        self.data_dir = data_dir
        self.shape = [self.read_batch, self.output_size, self.output_size, self.c_dim]
        self.coord = None
        self.threads = None
        self.format = format
        self.with_labels = with_labels
        if self.format == 'NCHW':
            self.shape = [self.read_batch,  self.c_dim, self.output_size, self.output_size]

    def _transform(self, x):
        return x

    def connect(self):
        assert hasattr(self, 'single_sample'), 'Pipeline needs to have single_sample defined before connecting'
        with tf.device('/cpu:0'):
            self.single_sample.set_shape(self.shape)
            ims = tf.train.shuffle_batch(
                [self.single_sample],
                self.batch_size,
                capacity=self.read_batch,
                min_after_dequeue=self.read_batch//8,
                num_threads=16,
                enqueue_many=len(self.shape) == 4)
            ims = self._transform(ims)
            images_shape = ims.get_shape()
            image_producer_stage = StagingArea(dtypes=[tf.float32], shapes=[images_shape])
            image_producer_op = image_producer_stage.put([ims])
            image_producer_stage_get = image_producer_stage.get()[0]
            images = tf.tuple([image_producer_stage_get], control_inputs=[image_producer_op])[0]
        return images

    def start(self, sess):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def stop(self):
        self.coord.request_stop()
        self.coord.join(self.threads)


class ConstantPipe(Pipeline):
    def __init__(self, *args, **kwargs):
        super(ConstantPipe, self).__init__(*args, **kwargs)
        stddev = 0.2
        if self.format == 'NCHW':
            out_shape = [self.batch_size, self.c_dim, self.output_size, self.output_size]
        elif self.format == 'NHWC':
            out_shape = [self.batch_size, self.output_size, self.output_size, self.c_dim]

        X = tf.get_variable('X', out_shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=False)
        self.images = X

    def connect(self):
        return self.images


class DataFlow(Pipeline):

    def __init__(self, *args, **kwargs):
        super(DataFlow, self).__init__(*args, **kwargs)
        self.pattern = 'tf_records_train/train*'

        cpu_device = '/cpu:0'

        # Preprocessing
        with tf.device(cpu_device):
            file_pattern = os.path.join(self.data_dir, self.pattern)
            record_input = RecordInput(
                file_pattern=file_pattern,
                seed=Record_seed,
                parallelism=32,
                buffer_size=4000,
                batch_size=self.batch_size,
                shift_ratio=0,
                name='record_input')
            records = record_input.get_yield_op()
            records = tf.split(records, self.batch_size, 0)
            records = [tf.reshape(record, []) for record in records]
            images = []
            labels = []
            for idx in xrange(self.batch_size):
                value = records[idx]
                if self.with_labels:
                    image, label = self.parse_example_proto_and_process(value)
                    labels.append(label)
                else:
                    image = self.parse_example_proto_and_process(value)
                images.append(image)
            if self.with_labels:
                labels = tf.parallel_stack(labels, 0)
                labels = tf.reshape(labels, [self.batch_size])

            images = tf.parallel_stack(images)
            images = tf.reshape(images, shape=[self.batch_size, self.output_size, self.output_size, self.c_dim])

            if self.format == 'NCHW':
                images = tf.transpose(images, [0, 3, 1, 2])
            images_shape = images.get_shape()
            if self.with_labels:
                labels_shape = labels.get_shape()
                image_producer_stage = StagingArea(dtypes=[tf.float32, tf.int32], shapes=[images_shape, labels_shape])
                image_producer_op = image_producer_stage.put([images, labels])
                image_producer_stage_get = image_producer_stage.get()
                images_and_labels = tf.tuple([image_producer_stage_get[0], image_producer_stage_get[1]], control_inputs=[image_producer_op])
                images = images_and_labels[0]
                labels = images_and_labels[1]
            else:
                image_producer_stage = StagingArea(dtypes=[tf.float32], shapes=[images_shape])
                image_producer_op = image_producer_stage.put([images])
                image_producer_stage_get = image_producer_stage.get()[0]
                images = tf.tuple([image_producer_stage_get], control_inputs=[image_producer_op])[0]

        self.images = images
        self.image_producer_op = image_producer_op
        if self.format == 'NCHW':
            self.shape = [self.c_dim, self.output_size, self.output_size]
        elif self.format == 'NHWC':
            self.shape = [self.output_size, self.output_size, self.c_dim]
        if self.with_labels:
            self.labels = labels
        #self.cpu_compute_stage_op = cpu_compute_stage_op

    def connect(self):
        if self.with_labels:
            return self.images, self.labels
        else:
            return self.images

    def start(self, sess):
        sess.run([self.image_producer_op])
        #sess.run([self.cpu_compute_stage_op])

    def stop(self):
        self.image_producer_op = None
        #self.cpu_compute_stage_op = None

    def parse_example_proto_and_process(self, value):
        raise NotImplementedError


class ImagenetDataFlow(DataFlow):

    def __init__(self, *args, **kwargs):
        super(ImagenetDataFlow, self).__init__(*args, **kwargs)

    def parse_example_proto_and_process(self, value):
        feature = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)}

        features = tf.parse_single_example(value, features=feature)
        #self.height = features['image/height']
        #self.width = features['image/width']
        image_buffer = features['image/encoded']

        image = self.preprocess(image_buffer)
        if self.with_labels:
            label = tf.cast(features['image/class/label'], dtype=tf.int32)
            return image, label
        else:
            return image

    def preprocess(self, image_buffer):
        image = tf.image.decode_jpeg(image_buffer, channels=self.c_dim)
        shape = [256, 256, 3]
        image = tf.reshape(image, shape)
        image = tf.cast(image, tf.float32)/255.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, (self.output_size, self.output_size))
        image = tf.squeeze(image, axis=0)

        return image


class CelebADataFlow(DataFlow):

    def __init__(self, *args, **kwargs):
        super(CelebADataFlow, self).__init__(*args, **kwargs)

    def parse_example_proto_and_process(self, value):
        feature = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)}

        features = tf.parse_single_example(value, features=feature)
        height = features['image/height']
        width = features['image/width']
        image_buffer = features['image/encoded']

        image = self.preprocess(image_buffer, height, width)

        return image

    def preprocess(self, image_buffer, height, width):
        image = tf.image.decode_jpeg(image_buffer, channels=self.c_dim)

        base_size = 160
        random_crop = 9
        bs = base_size + 2 * random_crop
        cropped = tf.image.resize_image_with_crop_or_pad(image, bs, bs)
        if random_crop > 0:
            cropped = tf.image.random_flip_left_right(cropped)
            cropped = tf.random_crop(cropped, [base_size, base_size, self.c_dim])
        image = cropped
        image = tf.cast(image, tf.float32)/255.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, (self.output_size, self.output_size))
        image = tf.squeeze(image, axis=0)

        return image


class LMDB(Pipeline):
    def __init__(self, timer=None, *args,  **kwargs):
#        print(*args)
#        print(**kwargs)
        super(LMDB, self).__init__(*args, **kwargs)
        self.timer = timer
        self.keys = []
        env = lmdb.open(self.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            while cursor.next():
                self.keys.append(cursor.key())
        print('Number of records in lmdb: %d' % len(self.keys))
        env.close()
        # tf queue for getting keys
        key_producer = tf.train.string_input_producer(self.keys, shuffle=True)
        single_key = key_producer.dequeue()
        self.single_sample = tf.py_func(self._get_sample_from_lmdb, [single_key], tf.float32)

    def _get_sample_from_lmdb(self, key, limit=None):
        if limit is None:
            limit = self.read_batch
        with tf.device('/cpu:0'):
            rc = self.read_count
            self.read_count += 1
            tt = time.time()
            self.timer(rc, 'read start')
            env = lmdb.open(self.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
            ims = []
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                cursor.set_key(key)
                while len(ims) < limit:
                    key, byte_arr = cursor.item()
                    byte_im = io.BytesIO(byte_arr)
                    byte_im.seek(0)
                    try:
                        im = Image.open(byte_im)
                        ims.append(misc.center_and_scale(im, size=self.output_size))
                    except Exception as e:
                        print(e)
                    if not cursor.next():
                        cursor.first()
            env.close()
            self.timer(rc, 'read time = %f' % (time.time() - tt))
            return np.asarray(ims, dtype=np.float32)

    def constant_sample(self, size):
        choice = np.random.choice(self.keys, 1)[0]
        return self._get_sample_from_lmdb(choice, limit=size)


class TfRecords(Pipeline):
    def __init__(self, *args, **kwargs):
        regex = os.path.join(self.data_dir, 'lsun-%d/bedroom_train_*' % self.output_size)
        filename_queue = tf.train.string_input_producer(tf.gfile.Glob(regex), num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'image/class/label': tf.FixedLenFeature([1], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
        })
        image = tf.image.decode_jpeg(features['image/encoded'])

        self.single_sample = tf.cast(image, tf.float32)/255.
        self.shape = [self.output_size, self.output_size, self.c_dim]


class JPEG(Pipeline):
    def __init__(self, *args,  **kwargs):
        super(JPEG, self).__init__(*args, **kwargs)
        base_size = 160
        random_crop = 9
        files = glob(os.path.join(self.data_dir, '*.jpg'))

        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        reader = tf.WholeFileReader()
        _, raw = reader.read(filename_queue)
        decoded = tf.image.decode_jpeg(raw, channels=self.c_dim)  # HWC
        bs = base_size + 2 * random_crop
        cropped = tf.image.resize_image_with_crop_or_pad(decoded, bs, bs)
        if random_crop > 0:
            cropped = tf.image.random_flip_left_right(cropped)
            cropped = tf.random_crop(cropped, [base_size, base_size, self.c_dim])
        self.single_sample = cropped
        self.shape = [base_size, base_size, self.c_dim]

    def _transform(self, x):
        x = tf.image.resize_bilinear(x, (self.output_size, self.output_size))
        if self.format == 'NCHW':
            x = tf.transpose(x, [0, 3, 1, 2])
        return tf.cast(x, tf.float32)/255.


class Mnist(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Mnist, self).__init__(*args, **kwargs)
        fd = open(os.path.join(self.data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if self.format == 'NCHW':
            trX = loaded[16:].reshape((60000, 1, 28, 28)).astype(np.float)
        elif self.format == 'NHWC':
            trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if self.format == 'NCHW':
            teX = loaded[16:].reshape((10000, 1, 28, 28)).astype(np.float)
        elif self.format == 'NHWC':
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)

        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


class Cifar10(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Cifar10, self).__init__(*args, **kwargs)
        self.categories = np.arange(10)

        batchesX, batchesY = [], []
        for batch in range(1, 6):
            pth = os.path.join(self.data_dir, 'data_batch_{}'.format(batch))
            labels, pixels = self.load_batch(pth)
            batchesX.append(pixels)
            batchesY.append(labels)
        trX = np.concatenate(batchesX, axis=0)

        _, teX = self.load_batch(os.path.join(self.data_dir, 'test_batch'))

        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)

        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)

    def load_batch(self, pth):
        if os.path.exists(pth):
            loaded = misc.unpickle(pth)
            labels = np.asarray(loaded['labels'])
            
            try: 
                pixels = np.asarray(loaded['pixels'])
            except:
                pixels = np.asarray(loaded['data'])
        elif os.path.exists(pth + '.bin'):
            loaded = np.fromfile(pth + '.bin', dtype=np.uint8).reshape(-1, 3073)
            labels = loaded[:, 0]
            pixels = loaded[:, 1:]
        else:
            raise ValueError("couldn't find {}".format(pth))

        idx = np.in1d(labels, self.categories)
        labels = labels[idx]
        pixels = pixels[idx].reshape(-1, 3, 32, 32)
        if self.format == 'NHWC':
            pixels = pixels.transpose(0, 2, 3, 1)
        return labels, pixels


class GaussianMix(Pipeline):
    def __init__(self, sample_dir='/', means=[.0, 3.0], stds=[1.0, .5], size=1000,  *args, **kwargs):
        super(GaussianMix, self).__init__(*args, **kwargs)
        from matplotlib import animation
        X_real = np.r_[
            np.random.normal(0,  1, size=size),
            np.random.normal(3, .5, size=size),
        ]
        X_real = X_real.reshape(X_real.shape[0], 1, 1, 1)

        xlo = -5
        xhi = 7

        ax1 = plt.gca()
        fig = ax1.figure
        ax1.grid(False)
        ax1.set_yticks([], [])
        myhist(X_real.ravel(), color='r')
        ax1.set_xlim(xlo, xhi)
        ax1.set_ylim(0, 1.05)
        ax1._autoscaleXon = ax1._autoscaleYon = False

        wrtr = animation.writers['ffmpeg'](fps=20)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        wrtr.setup(fig=fig, outfile=os.path.join(sample_dir, 'train.mp4'), dpi=100)
        self.G_config = {
            'g_line': None,
            'ax1': ax1,
            'writer': wrtr,
            'figure': ax1.figure}
        queue = tf.train.input_producer(tf.constant(X_real.astype(np.float32)), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


def myhist(X, ax=plt, bins='auto', **kwargs):
    hist, bin_edges = np.histogram(X, bins=bins)
    hist = hist / hist.max()
    return ax.plot(
        np.c_[bin_edges, bin_edges].ravel(),
        np.r_[0, np.c_[hist, hist].ravel(), 0],
        **kwargs
    )


def get_pipeline(dataset, info):
    if 'lsun' in dataset:
        if 'tf_records' in info:
            return TfRecords
        else:
            return LMDB
    if dataset == 'celebA':
        return CelebADataFlow
        #return JPEG
    if dataset == 'mnist':
        return Mnist
    if dataset == 'cifar10':
        return Cifar10
    if dataset == 'imagenet':
        return ImagenetDataFlow
    if dataset == 'GaussianMix':
        return GaussianMix
    else:
        raise Exception('invalid dataset: %s' % dataset)
