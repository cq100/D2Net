#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

class CFLAGS(): 
    def __init__(self):
        self.scale_v5 = 'yolov5l'
        self.size = 416
        self.weights =  'epoch-200.h5'        
        self.save_frequency = 5
        self.save_model_dir = "saved_model/"        
        self.annotation_path = "./data/dataset/train.txt"
        self.iou = 0.5
        self.score = 0.25
        self.score_thres = 0.2

    def weights(self):
        return self.weights
    def save_frequency(self):
        return self.save_frequency
    def save_model_dir(self):
        return self.save_model_dir
    def size(self):
        return self.size
    def annotation_path(self):
        return self.annotation_path
    def iou(self):
        return self.iou
    def score(self):
        return self.score
    def score_thres(self):
        return self.score_thres
    def framework(self):
        return self.framework
    def scale_v5(self):
        return self.scale_v5
    
__C                           = edict()


cfg                           = __C


__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/LED.names"
__C.YOLO.ANCHORS              = [0.02571906,0.02571906,0.04297523,0.04297523,0.05171064,0.05171064, 0.07998481,0.07998481, 0.12528986,0.12528986,0.18328535,0.18328535, 0.19022334,0.19022334, 0.3026311,0.3026311, 0.4593428,0.4593428]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5
__C.YOLO.V5_STR               = [[64],[128],[128,3],[256],[256,9],[512],[512,9],[1024],[1024],[1024,3],[512],[512,3],[256],[256,3],[256],[512,3],[512],[1024,3]]
__C.YOLO.V5_GDGW              = [[0.33,0.50],[0.67,0.75],[1.0,1.0],[1.33,1.25]]

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/train.txt"
__C.TRAIN.BATCH_SIZE          = 8
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.GRADNORM            = True
__C.TRAIN.DoubleGAN           = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 1
__C.TRAIN.FISRT_STAGE_EPOCHS    = 100
__C.TRAIN.SECOND_STAGE_EPOCHS   = 101



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/test.txt"
__C.TEST.BATCH_SIZE           = 1
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5


