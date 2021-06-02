#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from core.config import cfg

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def Focus(input_layer,filters_shape,name=None):
    input_layer = tf.concat([input_layer[:, ::2, ::2, :], input_layer[:, 1::2, ::2, :],input_layer[:, ::2, 1::2, :],input_layer[:, 1::2, 1::2, :]], axis=-1)
    return convolutional(input_layer, filters_shape,activate_type='SiLU',name=name)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky',kernel_regularizer1=tf.keras.regularizers.l2(0.0005),kernel_initializer1=tf.random_normal_initializer(stddev=0.01),name=None):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    if name != None:
        if name[0]==0:
            c_name = 'D_conv2d_%d' %name[1] if name[1] > 0 else 'D_conv2d'
            b_name = 'D_batch_normalization_%d' %name[2] if name[2] > 0 else 'D_batch_normalization'
        elif name[0]==1:
            c_name = 'G_conv2d_%d' %name[1] if name[1] > 0 else 'G_conv2d'
            b_name = 'G_batch_normalization_%d' %name[2] if name[2] > 0 else 'G_batch_normalization'
            
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=kernel_regularizer1,
                                  kernel_initializer=kernel_initializer1,
                                  bias_initializer=tf.constant_initializer(0.),name=c_name)(input_layer)

    if bn: conv = BatchNormalization(name=b_name)(conv)        
        
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
        elif activate_type == "SiLU":
            conv = tf.nn.silu(conv)
    return conv


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky',cut=True,rname=None):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type,name=[rname[0],rname[1],rname[1]])
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1,filter_num2), activate_type=activate_type,name=[rname[0],rname[2],rname[2]])
    if cut == True:
        residual_output = short_cut + conv
    else:
        residual_output = conv        
    return residual_output

def C3(input_layer, c1, c2, n, cut1=True,C3_name=None):

    x = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type="SiLU",name=[C3_name[0],C3_name[1],C3_name[1]])
    input_layer = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type="SiLU",name=[C3_name[0],C3_name[1]+1,C3_name[1]+1])

    Res_name = np.arange(C3_name[1]+2,C3_name[1]+3+(2*n),1)
    for i in range(n):
        input_layer = residual_block(input_layer,int(c2/2), int(c2/2), int(c2/2), activate_type="SiLU", cut=cut1,rname=[C3_name[0],Res_name[2*i],Res_name[2*i+1]])
    return convolutional(tf.concat([x,input_layer],axis=-1), (1, 1, c2, c2), activate_type="SiLU",name=[C3_name[0],Res_name[-1],Res_name[-1]])


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def CSP_CBAM(input_layer,activate_type1="SiLU",CB_name=None):
    if CB_name != None:
        name_num = np.arange(CB_name[1],CB_name[1]+10,1)                
    c1 = c2 = input_layer.shape[3]
    x = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[0],name_num[0]])
    input_layer = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[1],name_num[1]])

    short_cut = input_layer
    input_layer = convolutional(input_layer, filters_shape=(1, 1, int(c2/2), int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[2],name_num[2]])
    input_layer = convolutional(input_layer, filters_shape=(3, 3,int(c2/2),int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[3],name_num[3]])

    F_avg = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
    F_max = tf.keras.layers.GlobalMaxPooling2D()(input_layer)
    F_avg = tf.keras.layers.Reshape((1, 1, F_avg.shape[1]))(F_avg)  # shape (None, 1, 1 feature)
    F_max = tf.keras.layers.Reshape((1, 1, F_max.shape[1]))(F_max)

    F_avg = convolutional(F_avg, filters_shape=(1, 1,int(c2/2),int(c2/4)),bn=False,
                          activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                          name=[CB_name[0],name_num[4],name_num[4]])

    F_avg = convolutional(F_avg, filters_shape=(1, 1,int(c2/4),int(c2/2)),bn=False,
                          activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                          name=[CB_name[0],name_num[5],name_num[5]])

    F_max = convolutional(F_max, filters_shape=(1, 1,int(c2/2),int(c2/4)),bn=False,
                          activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                          name=[CB_name[0],name_num[6],name_num[6]])

    F_max = convolutional(F_max, filters_shape=(1, 1,int(c2/4),int(c2/2)),bn=False,
                          activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                          name=[CB_name[0],name_num[7],name_num[7]])

   
    Channel_Attention = input_layer * (tf.nn.sigmoid(F_avg+F_max))

    avg_out = tf.reduce_mean(Channel_Attention, axis=3)
    max_out = tf.reduce_max(Channel_Attention, axis=3)
    Spatial_factor = convolutional(tf.stack([avg_out, max_out], axis=3), filters_shape=(7, 7,2,1),bn=False,
                          activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                          kernel_initializer1='he_normal',         
                          name=[CB_name[0],name_num[8],name_num[8]])
    SpatialAttention = Channel_Attention * Spatial_factor
    input_layer = short_cut + SpatialAttention
    input_layer = convolutional(tf.concat([x,input_layer],axis=-1), (1, 1, c2, c2), activate_type=activate_type1,name=[CB_name[0],name_num[9],name_num[9]])
    return input_layer


def upsample(input_layer, activate_type='leaky',name=None):
    if name != None:
        if name[0]==0:
            c_name = 'D_conv2d_%d' %name[1] 
            b_name = 'D_batch_normalization_%d' %name[2] 
        elif name[0]==1:
            c_name = 'G_conv2d_%d' %name[1] 
            b_name = 'G_batch_normalization_%d' %name[2]
    if cfg.TRAIN.GRADNORM:
        conv = tf.keras.layers.Conv2DTranspose(input_layer.shape[3], 3,strides=(2, 2),padding='same',name=c_name)(input_layer)
        conv = BatchNormalization(name=b_name)(conv)
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
        elif activate_type == "SiLU":
            conv = tf.nn.silu(conv)
        return conv
    else:
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
        
    

