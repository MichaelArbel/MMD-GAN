#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:34:47 2018

@author: mikolajbinkowski
"""
import tensorflow as tf
from core.snops import batch_norm, conv2d, deconv2d, linear, lrelu, linear_one_hot
from utils.misc import conv_sizes
# Generators


class Generator(object):
    def __init__(self, dim, c_dim, output_size, use_batch_norm, prefix='g_', with_sn=False, scale=1.0, with_learnable_sn_scale=False, format='NCHW'):
        self.used = False
        self.use_batch_norm = use_batch_norm
        self.dim = dim
        self.c_dim = c_dim
        self.output_size = output_size
        self.prefix = prefix
        self.with_sn = with_sn
        self.scale = scale
        self.with_learnable_sn_scale = with_learnable_sn_scale
        self.format = format
        if use_batch_norm:
            self.g_bn0 = batch_norm(name=prefix + 'bn0', format=self.format)
            self.g_bn1 = batch_norm(name=prefix + 'bn1', format=self.format)
            self.g_bn2 = batch_norm(name=prefix + 'bn2', format=self.format)
            self.g_bn3 = batch_norm(name=prefix + 'bn3', format=self.format)
            self.g_bn4 = batch_norm(name=prefix + 'bn4', format=self.format)
            self.g_bn5 = batch_norm(name=prefix + 'bn5', format=self.format)
        else:
            self.g_bn0 = lambda x: x
            self.g_bn1 = lambda x: x
            self.g_bn2 = lambda x: x
            self.g_bn3 = lambda x: x
            self.g_bn4 = lambda x: x
            self.g_bn5 = lambda x: x

    def __call__(self, seed, batch_size, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('generator') as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            return self.network(seed, batch_size, update_collection)

    def network(self, seed, batch_size, update_collection):
        pass

    def data_format(self, batch_size, height, width, channel):
        if self.format == 'NCHW':
            return [batch_size, channel, height, width]
        elif self.format == 'NHWC':
            return [batch_size, height, width, channel]


class CondGenerator(Generator):
    def __init__(self,  num_classes, *args, **kwargs):
        self.num_classes = num_classes
        super(CondGenerator, self).__init__(*args, **kwargs)

    def __call__(self, seed, y, batch_size, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('generator') as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            return self.network(seed, y, batch_size, update_collection)


class CondSNResNetGenerator(CondGenerator):
    def network(self, seed, y, batch_size, update_collection):
        from core.resnet import block, ops
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        if self.output_size == 64:
            s32 = 4

        z_ = linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin')
        h0 = tf.reshape(z_, [-1, self.dim * 16, s32, s32])  # NCHW format
        if self.output_size == 64:
            h0_bis = h0
        else:
            h0_bis = block.ResidualBlock(self.prefix + 'res0_bis', 16 * self.dim,
                                         16 * self.dim, 3, h0, y=y, num_classes=self.num_classes, resample='up', mode='cond_batchnorm')
        h1 = block.ResidualBlock(self.prefix + 'res1', 16 * self.dim,
                                 8 * self.dim, 3, h0_bis, y=y, num_classes=self.num_classes, resample='up', mode='cond_batchnorm')
        h2 = block.ResidualBlock(self.prefix + 'res2', 8 * self.dim,
                                 4 * self.dim, 3, h1, y=y, num_classes=self.num_classes, resample='up', mode='cond_batchnorm')
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim,
                                 2 * self.dim, 3, h2, y=y, num_classes=self.num_classes, resample='up', mode='cond_batchnorm')
        h4 = block.ResidualBlock(self.prefix + 'res4', 2 * self.dim,
                                 self.dim, 3, h3, y=y, num_classes=self.num_classes, resample='up', mode='cond_batchnorm')

        h4 = ops.batchnorm.Batchnorm('g_h4', [0, 2, 3], h4)
        h4 = tf.nn.relu(h4)
        if self.format == 'NHWC':
            h4 = tf.transpose(h4, [0, 2, 3, 1])  # NCHW to NHWC
        h5 = deconv2d(h4, self.data_format(batch_size, s1, s1, self.c_dim), k_h=3, k_w=3, d_h=1, d_w=1, name=self.prefix+'g_h5')
        return tf.nn.sigmoid(h5)


class DCGANGenerator(Generator):
    def network(self, seed, batch_size, update_collection):
        s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
        # 64, 32, 16, 8, 4 - for self.output_size = 64
        # default architecture
        # For Cramer: self.gf_dim = 64
        z_ = linear(seed, self.dim * 8 * s16 * s16, self.prefix + 'h0_lin', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale)   # project random noise seed and reshape

        h0 = tf.reshape(z_, self.data_format(batch_size, s16, s16, self.dim * 8))
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = deconv2d(h0, self.data_format(batch_size, s8, s8, self.dim*4), name=self.prefix + 'h1', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, self.data_format(batch_size, s4, s4, self.dim*2), name=self.prefix + 'h2', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, self.data_format(batch_size, s2, s2, self.dim*1), name=self.prefix + 'h3', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = deconv2d(h3, self.data_format(batch_size, s1, s1, self.c_dim), name=self.prefix + 'h4', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        return tf.nn.sigmoid(h4)


class DCGAN5Generator(Generator):
    def network(self, seed, batch_size, update_collection):
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        z_ = linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale)

        h0 = tf.reshape(z_, self.data_format(-1, s32, s32, self.dim * 16))
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = deconv2d(h0, self.data_format(batch_size, s16, s16, self.dim*8), name=self.prefix + 'h1', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, self.data_format(batch_size, s8, s8, self.dim*4), name=self.prefix + 'h2', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, self.data_format(batch_size, s4, s4, self.dim*2), name=self.prefix + 'h3', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = deconv2d(h3, self.data_format(batch_size, s2, s2, self.dim), name=self.prefix + 'h4', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h4 = tf.nn.relu(self.g_bn4(h4))

        h5 = deconv2d(h4, self.data_format(batch_size, s1, s1, self.c_dim), name=self.prefix + 'h5', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        return tf.nn.sigmoid(h5)


class ResNetGenerator(Generator):
    def network(self, seed, batch_size, update_collection):
        from core.resnet import block, ops
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        z_ = linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin')
        h0 = tf.reshape(z_, [-1, self.dim * 16, s32, s32])  # NCHW format

        h1 = block.ResidualBlock(self.prefix + 'res1', 16 * self.dim,
                                 8 * self.dim, 3, h0, resample='up')
        h2 = block.ResidualBlock(self.prefix + 'res2', 8 * self.dim,
                                 4 * self.dim, 3, h1, resample='up')
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim,
                                 2 * self.dim, 3, h2, resample='up')
        h4 = block.ResidualBlock(self.prefix + 'res4', 2 * self.dim,
                                 self.dim, 3, h3, resample='up')

        h4 = ops.batchnorm.Batchnorm('g_h4', [0, 2, 3], h4)
        h4 = tf.nn.relu(h4)
#                h5 = lib.ops.conv2d.Conv2D('g_h5', dim, 3, 3, h4)
        if self.format == 'NHWC':
            h4 = tf.transpose(h4, [0, 2, 3, 1])  # NCHW to NHWC
        h5 = deconv2d(h4, self.data_format(batch_size, s1, s1, self.c_dim), name=self.prefix + 'g_h5')
        return tf.nn.sigmoid(h5)


class SNResNetGenerator(Generator):
    def network(self, seed, batch_size, update_collection):
        from core.resnet import block, ops
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        if self.output_size == 64:
            s32 = 4

        z_ = linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin')
        h0 = tf.reshape(z_, [-1, self.dim * 16, s32, s32])  # NCHW format
        if self.output_size == 64:
            h0_bis = h0
        else:
            h0_bis = block.ResidualBlock(self.prefix + 'res0_bis', 16 * self.dim,
                                         16 * self.dim, 3, h0, resample='up', mode='batchnorm')
        h1 = block.ResidualBlock(self.prefix + 'res1', 16 * self.dim,
                                 8 * self.dim, 3, h0_bis, resample='up', mode='batchnorm')
        h2 = block.ResidualBlock(self.prefix + 'res2', 8 * self.dim,
                                 4 * self.dim, 3, h1, resample='up', mode='batchnorm')
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim,
                                 2 * self.dim, 3, h2, resample='up', mode='batchnorm')
        h4 = block.ResidualBlock(self.prefix + 'res4', 2 * self.dim,
                                 self.dim, 3, h3, resample='up', mode='batchnorm')

        h4 = ops.batchnorm.Batchnorm('g_h4', [0, 2, 3], h4)
        h4 = tf.nn.relu(h4)
#                h5 = lib.ops.conv2d.Conv2D('g_h5', dim, 3, 3, h4)
        if self.format == 'NHWC':
            h4 = tf.transpose(h4, [0, 2, 3, 1])  # NCHW to NHWC
        h5 = deconv2d(h4, self.data_format(batch_size, s1, s1, self.c_dim), k_h=3, k_w=3, d_h=1, d_w=1, name=self.prefix+'g_h5')
        return tf.nn.sigmoid(h5)


class SNGANGenerator(Generator):
    # DCGAN Generator used in 'Spectral Normalization in GANs', based on https://github.com/minhnhat93/tf-SNDCGAN/blob/master/net.py
    def network(self, seed, batch_size, update_collection):
        s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
        z_ = linear(seed, self.dim * 8 * s8 * s8, self.prefix + 'h0_lin', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale)  # project random noise seed and reshape

        h0 = tf.reshape(z_, self.data_format(batch_size, s8, s8, self.dim * 8))
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = deconv2d(h0, self.data_format(batch_size, s4, s4, self.dim*4), name=self.prefix + 'h1', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, self.data_format(batch_size, s2, s2, self.dim*2), name=self.prefix + 'h2', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, self.data_format(batch_size, s1, s1, self.dim*1), name=self.prefix + 'h3', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        h3 = tf.nn.relu(self.g_bn3(h3))
        # SN dcgan generator implementation has smaller convolutional field and stride=1
        h4 = deconv2d(h3, self.data_format(batch_size, s1, s1, self.c_dim), k_h=3, k_w=3, d_h=1, d_w=1, name=self.prefix + 'h4', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format)
        return tf.nn.sigmoid(h4)


# Discriminator

class Discriminator(object):
    def __init__(self, dim, o_dim, use_batch_norm, prefix='d_', with_sn=False, scale=1.0, with_learnable_sn_scale=False, format='NCHW'):
        self.dim = dim
        self.o_dim = o_dim
        self.prefix = prefix
        self.used = False
        self.use_batch_norm = use_batch_norm
        self.with_sn = with_sn
        self.scale = scale
        self.with_learnable_sn_scale = with_learnable_sn_scale
        self.format = format
        if use_batch_norm:
            self.d_bn0 = batch_norm(name=prefix + 'bn0', format=self.format)
            self.d_bn1 = batch_norm(name=prefix + 'bn1', format=self.format)
            self.d_bn2 = batch_norm(name=prefix + 'bn2', format=self.format)
            self.d_bn3 = batch_norm(name=prefix + 'bn3', format=self.format)
            self.d_bn4 = batch_norm(name=prefix + 'bn4', format=self.format)
            self.d_bn5 = batch_norm(name=prefix + 'bn5', format=self.format)
        else:
            self.d_bn0 = lambda x: x
            self.d_bn1 = lambda x: x
            self.d_bn2 = lambda x: x
            self.d_bn3 = lambda x: x
            self.d_bn4 = lambda x: x
            self.d_bn5 = lambda x: x

    def __call__(self, image, batch_size, return_layers=False,  update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope("discriminator") as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            layers = self.network(image, batch_size, update_collection)
            if return_layers:
                return layers
            return layers['hF']

    def network(self, image, batch_size):
        pass


class CondDiscriminator(Discriminator):
    def __init__(self,  num_classes, *args, **kwargs):
        self.num_classes = num_classes
        super(CondDiscriminator, self).__init__(*args, **kwargs)

    def __call__(self, seed, batch_size, return_layers=False, update_collection=tf.GraphKeys.UPDATE_OPS, y=None):
        with tf.variable_scope('discriminator') as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            layers = self.network(seed, batch_size, update_collection, y)
            if return_layers:
                return layers
            return layers['hF']


class CondProjectionSNResNetDiscriminator(CondDiscriminator):
    def network(self, image, batch_size, update_collection, y):
        from core.resnet import block, ops
        if self.format == 'NHWC':
            image = tf.transpose(image, [0, 3, 1, 2])  # NHWC to NCHW
        h0 = lrelu(ops.conv2d.Conv2D(self.prefix + 'h0_conv', 3, self.dim,
                                     3, image, update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale))
        h1 = block.ResidualBlock(self.prefix + 'res1', self.dim,
                                 2 * self.dim, 3, h0, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        h2 = block.ResidualBlock(self.prefix + 'res2', 2 * self.dim,
                                 4 * self.dim, 3, h1, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim,
                                 8 * self.dim, 3, h2, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        h4 = block.ResidualBlock(self.prefix + 'res4', 8 * self.dim,
                                 16 * self.dim, 3, h3, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        if image.get_shape().as_list()[2] == 64:
            h4_bis = h4
        else:
            h4_bis = block.ResidualBlock(self.prefix + 'res4_bis', 16 * self.dim,
                                         16 * self.dim, 3, h4, resample=None, update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)

        h4_bis = lrelu(h4_bis)
        h4_bis = tf.reduce_sum(h4_bis, axis=[2, 3])
        hF = linear(h4_bis, self.o_dim, self.prefix + 'h5_lin', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        if not y is None:
            w_y = linear_one_hot(y, self.o_dim, self.num_classes, name=self.prefix+"Linear_one_hot", update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)

            hF += tf.reduce_sum(w_y*hF, axis=1, keepdims=True)

        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'hF': hF}


class DCGANDiscriminator(Discriminator):
    def network(self, image, batch_size, update_collection):
        o_dim = self.o_dim if (self.o_dim > 0) else 8 * self.dim
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        hF = linear(tf.reshape(h3, [batch_size, -1]), o_dim, self.prefix + 'h4_lin', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale)
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF}


class DCGAN5Discriminator(Discriminator):
    def network(self, image, batch_size, update_collection):
        o_dim = self.o_dim if (self.o_dim > 0) else 16 * self.dim
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.dim * 16, name=self.prefix + 'h4_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        hF = linear(tf.reshape(h4, [batch_size, -1]), o_dim, self.prefix + 'h6_lin', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale)
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'hF': hF}


class FullConvDiscriminator(Discriminator):
    def network(self, image, batch_size, update_collection):
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale, data_format=self.format,with_singular_values=True)))
        hF = lrelu(self.d_bn4(conv2d(h3, self.o_dim, name=self.prefix + 'hF_conv', update_collection=update_collection, with_sn=self.with_sn, scale=self.scale, with_learnable_sn_scale=self.with_learnable_sn_scale,with_singular_values=True)))
        hF = tf.reshape(hF, [batch_size, -1])
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF}


# Warning this implementation doesn't allow spectral normalization
class ResNetDiscriminator(Discriminator):
    def network(self, image, batch_size, update_collection):
        from core.resnet import block, ops
        if self.format == 'NHWC':
            image = tf.transpose(image, [0, 3, 1, 2])  # NHWC to NCHW
        h0 = lrelu(ops.conv2d.Conv2D(self.prefix + 'h0_conv', 3, self.dim,
                                     3, image))
        h1 = block.ResidualBlock(self.prefix + 'res1', self.dim,
                                 2 * self.dim, 3, h0, resample='down')
        h2 = block.ResidualBlock(self.prefix + 'res2', 2 * self.dim,
                                 4 * self.dim, 3, h1, resample='down')
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim,
                                 8 * self.dim, 3, h2, resample='down')
        h4 = block.ResidualBlock(self.prefix + 'res4', 8 * self.dim,
                                 8 * self.dim, 3, h3, resample='down')
        h4 = tf.reshape(h4, [-1, 4 * 4 * 8 * self.dim])
        hF = linear(h4, self.o_dim, self.prefix + 'h5_lin')
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'hF': hF}


class InjectiveDiscriminator(Discriminator):
    def __init__(self, net):
        self.net = net
        self.scale_id_layer = 1.
        super(InjectiveDiscriminator, self).__init__(net.dim, net.o_dim, net.use_batch_norm, prefix=net.prefix)

    def network(self, image, batch_size, update_collection):
        layers = self.net.network(image, batch_size)
        id_layer_0 = tf.reshape(image, [batch_size, -1])
        init_value = 1./(id_layer_0.get_shape().as_list()[-1])
        self.scale_id_layer = tf.get_variable(name=self.prefix+'scale_id_layer', shape=[1], initializer=tf.constant_initializer(init_value), trainable=True, dtype=tf.float32)
        id_layer = id_layer_0*self.scale_id_layer
        hF = tf.concat([layers['hF'], id_layer], 1)
        layers['hF'] = hF
        return layers


class SNGANDiscriminator(Discriminator):
    # Discriminator used in 'Spectral Normalization in GANs', based on https://github.com/minhnhat93/tf-SNDCGAN/blob/master/net.py
    def network(self, image, batch_size, update_collection):
        c0_0 = lrelu(conv2d(image, 64, 3, 3, 1, 1, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c0_0', data_format=self.format,with_singular_values=True))
        c0_1 = lrelu(conv2d(c0_0, 128, 4, 4, 2, 2, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c0_1', data_format=self.format,with_singular_values=True))
        c1_0 = lrelu(conv2d(c0_1, 128, 3, 3, 1, 1, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c1_0', data_format=self.format,with_singular_values=True))
        c1_1 = lrelu(conv2d(c1_0, 256, 4, 4, 2, 2, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c1_1', data_format=self.format,with_singular_values=True))
        c2_0 = lrelu(conv2d(c1_1, 256, 3, 3, 1, 1, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c2_0', data_format=self.format,with_singular_values=True))
        c2_1 = lrelu(conv2d(c2_0, 512, 4, 4, 2, 2, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c2_1', data_format=self.format,with_singular_values=True))
        c3_0 = lrelu(conv2d(c2_1, 512, 3, 3, 1, 1, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c3_0', data_format=self.format,with_singular_values=True))
        c3_0 = tf.reshape(c3_0, [batch_size, -1])
        l4 = linear(c3_0, self.o_dim, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale, update_collection=update_collection, stddev=0.02, name=self.prefix + 'l4')
        return {'h0': c0_0, 'h1': c0_1, 'h2': c1_0, 'h3': c1_1, 'h4': c2_0, 'h5': c2_1, 'h6': c3_0, 'hF': l4}


class SNResNetDiscriminator(Discriminator):
    def network(self, image, batch_size, update_collection):
        from core.resnet import block, ops
        if self.format == 'NHWC':
            image = tf.transpose(image, [0, 3, 1, 2])  # NHWC to NCHW
        h0 = lrelu(ops.conv2d.Conv2D(self.prefix + 'h0_conv', 3, self.dim,
                                     3, image, update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale))
        h1 = block.ResidualBlock(self.prefix + 'res1', self.dim,
                                 2 * self.dim, 3, h0, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        h2 = block.ResidualBlock(self.prefix + 'res2', 2 * self.dim,
                                 4 * self.dim, 3, h1, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim,
                                 8 * self.dim, 3, h2, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        h4 = block.ResidualBlock(self.prefix + 'res4', 8 * self.dim,
                                 16 * self.dim, 3, h3, resample='down', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        if image.get_shape().as_list()[2] == 64:
            h4_bis = h4
        else:
            h4_bis = block.ResidualBlock(self.prefix + 'res4_bis', 16 * self.dim,
                                         16 * self.dim, 3, h4, resample=None, update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)

        h4_bis = lrelu(h4_bis)
        h4_bis = tf.reduce_sum(h4_bis, axis=[2, 3])
        hF = linear(h4_bis, self.o_dim, self.prefix + 'h5_lin', update_collection=update_collection, with_sn=self.with_sn, with_learnable_sn_scale=self.with_learnable_sn_scale)
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'hF': hF}


def get_networks(architecture):
    print('architec', architecture)
    if architecture == 'dcgan':
        return DCGANGenerator, DCGANDiscriminator
    elif architecture == 'dcgan5':
        return DCGAN5Generator, DCGAN5Discriminator
    elif architecture == 'sngan':
        return SNGANGenerator, SNGANDiscriminator
    elif architecture == 'sngan-dcgan5':
        return SNGANGenerator, DCGAN5Discriminator
    elif architecture == 'snresnet':
        return SNResNetGenerator, SNResNetDiscriminator
    elif architecture == 'cond_snresnet':
        return CondSNResNetGenerator, CondProjectionSNResNetDiscriminator
    elif 'g-resnet5' in architecture:
        print('get_networks')
        print(ResNetGenerator, DCGAN5Discriminator)
        return ResNetGenerator, DCGAN5Discriminator
    elif architecture == 'resnet5':
        return ResNetGenerator, ResNetDiscriminator
    elif architecture == 'd-fullconv5':
        return DCGAN5Generator, FullConvDiscriminator
    raise ValueError('Wrong architecture: "%s"' % architecture)
