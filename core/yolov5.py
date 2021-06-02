##! /usr/bin/env python
## coding=utf-8
import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg
from functools import partial

NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
STRIDES  = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
XYSCALE = cfg.YOLO.XYSCALE
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)


class NLayerDiscriminator(tf.keras.Model):
    def __init__(self, ndf, n_layers):
        super(NLayerDiscriminator, self).__init__()

        self.kernel_size = 4
        self.padding_size = int(np.ceil((self.kernel_size - 1) / 2))
        self.initial = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=ndf, kernel_size=self.kernel_size, strides=2
                ),
                tf.keras.layers.LeakyReLU(0.2),
            ]
        )

        self.filters_upscale_blocks = []
        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            self.filters_upscale_blocks.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(
                            ndf * nf_mult, kernel_size=self.kernel_size, strides=2
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(0.2),
                    ]
                )
            )

        nf_mult = min(2 ** n_layers, 8)
        self.filters_upscale_blocks.append(
            tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        ndf * nf_mult,
                        kernel_size=self.kernel_size,
                        strides=1,
                        padding="valid",
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(0.2),
                ]
            )
        )

        self.final = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    1, kernel_size=self.kernel_size, strides=1, padding="valid"
                )
            ]
        )

        self.pad = partial(
            tf.pad,
            paddings=tf.constant(
                [
                    [0, 0],
                    [self.padding_size, self.padding_size],
                    [self.padding_size, self.padding_size],
                    [0, 0],
                ]
            ),
            mode="REFLECT",
        )

    def call(self, x):
        x = self.pad(x)
        x = self.initial(x)

        for upscale_filter_block in self.filters_upscale_blocks:
            x = self.pad(x)
            x = upscale_filter_block(x)

        x = self.pad(x)
        x = self.final(x)
        return tf.sigmoid(x)
##        return x
    

def YOLO(scale_v5, input_layer, NUM_CLASS):    
    return YOLOv5(input_layer, NUM_CLASS, scale_v5)

def YOLOv5(input_layer, NUM_CLASS, scale_v5):
    V5_STR  = utils.get_V5STR(scale_v5)
    print("YOLOv5 structure parameters:",V5_STR)
    N_start_next, route_1, route_2, route_3, G1, G2 = backbone.Focus_cspdarknet53(input_layer,V5_STR)

    ###Detection branch###
    G3, G4 = route_1, route_2
    route_3 = common.convolutional(route_3, (1, 1, V5_STR[9][0], V5_STR[10][0]),activate_type="SiLU",name=[0,N_start_next,N_start_next])
    R1 = route_3
    route_3 = common.upsample(route_3,activate_type="SiLU",name=[0,N_start_next+1,N_start_next+1])
    route_3 = tf.concat([route_2, route_3], axis=-1)
    route_3 = common.C3(route_3, V5_STR[10][0]*2, V5_STR[11][0], V5_STR[11][1], cut1=False,C3_name=[0,N_start_next+2])

    N_start_next = N_start_next+5+2*V5_STR[11][1]
    route_3 = common.convolutional(route_3, (1, 1, V5_STR[11][0], V5_STR[12][0]),activate_type="SiLU", name=[0,N_start_next,N_start_next])
    R2 = route_3
    route_3 = common.upsample(route_3,activate_type="SiLU", name=[0,N_start_next+1,N_start_next+1])
    route_3 = tf.concat([route_1, route_3], axis=-1)
    route_3 = common.C3(route_3, V5_STR[12][0]*2, V5_STR[13][0], V5_STR[13][1], cut1=False,C3_name=[0,N_start_next+2])
    N_start_next = N_start_next+5+2*V5_STR[13][1]
    conv_sbbox = common.convolutional(route_3, (1, 1, V5_STR[13][0], 3 * (NUM_CLASS + 5)), activate=False, bn=False, name=[0,N_start_next,N_start_next])

    route_3 = common.convolutional(route_3, (3, 3, V5_STR[13][0], V5_STR[14][0]), downsample=True, activate_type="SiLU", name=[0,N_start_next+1,N_start_next+1])
    route_3 = tf.concat([R2, route_3], axis=-1)
    route_3 = common.C3(route_3, V5_STR[14][0]*2, V5_STR[15][0], V5_STR[15][1], cut1=False,C3_name=[0,N_start_next+2])
    N_start_next = N_start_next+5+2*V5_STR[15][1]
    conv_mbbox = common.convolutional(route_3, (1, 1, V5_STR[15][0], 3 * (NUM_CLASS + 5)), activate=False, bn=False, name=[0,N_start_next,N_start_next])

    route_3 = common.convolutional(route_3, (3, 3, V5_STR[15][0], V5_STR[16][0]), downsample=True, activate_type="SiLU", name=[0,N_start_next+1,N_start_next+1])
    route_3 = tf.concat([R1, route_3], axis=-1)
    route_3 = common.C3(route_3, V5_STR[16][0]*2, V5_STR[17][0],V5_STR[17][1], cut1=False, C3_name=[0,N_start_next+2])
    N_start_next = N_start_next+5+2*V5_STR[17][1]
    conv_lbbox = common.convolutional(route_3, (1, 1, V5_STR[17][0], 3 * (NUM_CLASS + 5)), activate=False, bn=False,name=[0,N_start_next,N_start_next])

    ###Deblurring branch###
    out2 = common.convolutional(G2, (1, 1, 128, 128), activate=False, bn=False, name=[1,N_start_next+1,N_start_next+1])
    out2 = common.CSP_CBAM(out2, activate_type1="SiLU", CB_name=[1,N_start_next+2])
    N_start_next = N_start_next+12

    out3 = common.convolutional(G3, (1, 1, 256, 128), activate=False, bn=False, name=[1,N_start_next,N_start_next])
    out3 = common.CSP_CBAM(out3, activate_type1="SiLU", CB_name=[1,N_start_next+1])
    N_start_next = N_start_next+11
    
    out2 = tf.concat([common.upsample(out3,activate_type="SiLU",name=[1,N_start_next,N_start_next]), out2], axis=-1)
    out2 = common.convolutional(common.upsample(out2,activate_type="SiLU",name=[1,N_start_next+1,N_start_next+1]), (3, 3, 256, 64),activate_type="SiLU",name=[1,N_start_next+2,N_start_next+2])
    N_start_next = N_start_next+3
    
    out4 = common.convolutional(G4, (1, 1, 512, 128), activate=False, bn=False,name=[1,N_start_next,N_start_next])
    out4 = common.CSP_CBAM(out4, activate_type1="SiLU", CB_name=[1,N_start_next+1])
    N_start_next = N_start_next+11

    out3 = tf.concat([common.upsample(out4,activate_type="SiLU",name=[1,N_start_next,N_start_next]), out3], axis=-1)
    out3 = common.convolutional(common.upsample(out3,activate_type="SiLU",name=[1,N_start_next+1,N_start_next+1]), (3, 3, 256, 128),activate_type="SiLU",name=[1,N_start_next+2,N_start_next+2])
    out3 = common.convolutional(common.upsample(out3,activate_type="SiLU",name=[1,N_start_next+3,N_start_next+3]), (3, 3, 128, 64),activate_type="SiLU",name=[1,N_start_next+4,N_start_next+4])
    N_start_next = N_start_next+5

    out4 = common.convolutional(common.upsample(out4,activate_type="SiLU",name=[1,N_start_next,N_start_next]), (3, 3, 128, 128),activate_type="SiLU",name=[1,N_start_next+1,N_start_next+1])
    out4 = common.convolutional(common.upsample(out4,activate_type="SiLU",name=[1,N_start_next+2,N_start_next+2]), (3, 3, 128, 64),activate_type="SiLU",name=[1,N_start_next+3,N_start_next+3])
    out4 = common.convolutional(common.upsample(out4,activate_type="SiLU",name=[1,N_start_next+4,N_start_next+4]), (3, 3, 64, 64),activate_type="SiLU",name=[1,N_start_next+5,N_start_next+5])
    N_start_next = N_start_next+6
    
    concat = tf.concat([out2, out3, out4], axis=-1)
    concat = common.convolutional(concat, (3, 3, 64*3, 128),activate_type="SiLU",name=[1,N_start_next,N_start_next])
    
    out1 = common.convolutional(G1, (1, 1, 64, 128),activate=False, bn=False,name=[1,N_start_next+1,N_start_next+1])
    out1 = common.CSP_CBAM(out1, activate_type1="SiLU", CB_name=[1,N_start_next+2])
    N_start_next = N_start_next+12
    
    out1 = tf.concat([out1, concat], axis=-1)
    out1 = common.convolutional(common.upsample(out1,activate_type="SiLU",name=[1,N_start_next,N_start_next]), (3, 3, 256, 32),activate_type="SiLU",name=[1,N_start_next+1,N_start_next+1])
    out1 = common.convolutional(out1, (1, 1, 32, 3), activate=False, bn=False,name=[1,N_start_next+2,N_start_next+2])
    out1 = tf.math.sigmoid(out1) + input_layer
    result = tf.clip_by_value(out1, clip_value_min=0, clip_value_max=1)
    return [conv_sbbox, conv_mbbox, conv_lbbox, result]

def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1,1,1]):
    return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2) 
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([cfg.TRAIN.INPUT_SIZE,cfg.TRAIN.INPUT_SIZE])):
    scores_max = tf.math.reduce_max(scores, axis=-1)
    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)    
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  
        box_mins[..., 1:2],  
        box_maxes[..., 0:1],  
        box_maxes[..., 1:2]  
    ], axis=-1)
    return (boxes, pred_conf)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

##    iou_loss = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    iou_loss = tf.expand_dims(utils.bbox_diou(pred_xywh, label_xywh), axis=-1)
    
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    iou_loss = respond_bbox * bbox_loss_scale * (1- iou_loss)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    return iou_loss, conf_loss, prob_loss




