#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import core.common as common
                
def Focus_cspdarknet53(input_data,V5_STR):  
    input_data = common.Focus(input_data, (3, 3, 12, V5_STR[0][0]),name=[0,0,0])
    G1 = input_data
    input_data = common.convolutional(input_data, (3, 3, V5_STR[0][0], V5_STR[1][0]), downsample=True, activate_type="SiLU",name=[0,1,1])
    input_data = common.C3(input_data, V5_STR[1][0], V5_STR[2][0], V5_STR[2][1],C3_name=[0,2])

    N_start_next = 5 + V5_STR[2][1]*2
    G2 = input_data
    input_data = common.convolutional(input_data, (3, 3, V5_STR[2][0], V5_STR[3][0]), downsample=True, activate_type="SiLU",name=[0,N_start_next,N_start_next])
    input_data = common.C3(input_data, V5_STR[3][0], V5_STR[4][0], V5_STR[4][1],C3_name=[0,N_start_next+1])

    N_start_next = N_start_next+4+V5_STR[4][1]*2
    route1 = input_data 
    input_data = common.convolutional(input_data, (3, 3, V5_STR[4][0], V5_STR[5][0]), downsample=True, activate_type="SiLU",name=[0,N_start_next,N_start_next])
    input_data = common.C3(input_data, V5_STR[5][0], V5_STR[6][0], V5_STR[6][1],C3_name=[0,N_start_next+1])

    N_start_next = N_start_next+4+V5_STR[6][1]*2
    route2 = input_data 
    input_data = common.convolutional(input_data, (3, 3, V5_STR[6][0], V5_STR[7][0]), downsample=True, activate_type="SiLU",name=[0,N_start_next,N_start_next])
    ###SPP###
    input_data = common.convolutional(input_data, (1, 1, V5_STR[7][0], int(V5_STR[8][0]/2)),activate_type="SiLU",name=[0,N_start_next+1,N_start_next+1])
    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, V5_STR[8][0]*4, V5_STR[8][0]), activate_type="SiLU",name=[0,N_start_next+2,N_start_next+2])

    input_data = common.C3(input_data, V5_STR[8][0], V5_STR[9][0], V5_STR[9][1], cut1=False,C3_name=[0,N_start_next+3])
    route3 = input_data
    N_start_next = N_start_next+6+V5_STR[9][1]*2
    return N_start_next, route1, route2, route3, G1, G2
    
    







