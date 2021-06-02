#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


class Dataset(object):
    """implement Dataset here"""

    def __init__(self, FLAGS, is_training: bool, dataset_type: str = "converted_coco"):
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type
        self.N_planes = self.anchors.shape[0]
        
        self.annot_path = (
            cfg.TRAIN.ANNOT_PATH if is_training else cfg.TEST.ANNOT_PATH    
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG 

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE   
        self.max_bbox_per_scale = 50

        self.annotations = self.load_annotations() 
        self.num_samples = len(self.annotations)
        print("self.num_samples:",self.num_samples)    
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip()
                    for line in txt
                    if len(line.strip().split()[1:]) != 0
                ]
            elif self.dataset_type == "yolo":
                annotations = []
                for line in txt:
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd:
                        boxes = fd.readlines()
                        string = ""
                        for box in boxes:
                            box = box.strip()
                            box = box.split()
                            class_num = int(box[0])
                            center_x = float(box[1])
                            center_y = float(box[2])
                            half_width = float(box[3]) / 2
                            half_height = float(box[4]) / 2
                            string += " {},{},{},{},{}".format(
                                center_x - half_width,
                                center_y - half_height,
                                center_x + half_width,
                                center_y + half_height,
                                class_num,
                            )
                        annotations.append(image_path + string)

        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):                 
        with tf.device("/cpu:0"):
            # self.train_input_size = random.choice(self.train_input_sizes)
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides

            blur_batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )

            sharp_batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = np.zeros(                      
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )

            batch_sbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32                   )
            out_bboxes = []

            num = 0                                         
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:                
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    blur_image, sharp_image, bboxes = self.parse_annotation(annotation)
         
                    (
                    label_sbbox,
                    label_mbbox,
                    label_lbbox,
                    sbboxes,
                    mbboxes,
                    lbboxes,
                       ) = self.preprocess_true_boxes(bboxes)

                    blur_batch_image[num, :, :, :] = blur_image
                    sharp_batch_image[num, :, :, :] = sharp_image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes        
                        
                    num += 1

                    
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes  
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes
                
                return (
                (blur_batch_image,sharp_batch_image),
                (
                    batch_smaller_target,
                    batch_medium_target,
                    batch_larger_target,
                ),
                        )
                                
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)   
                raise StopIteration


    def random_horizontal_flip(self, blur_image, sharp_image, bboxes):
        if random.random() < 0.5:
            _, w1, _ = blur_image.shape
            blur_image = blur_image[:, ::-1, :]
            _, w2, _ = sharp_image.shape
            sharp_image = sharp_image[:, ::-1, :]
            if w1 != w2:
                print("images do not match...")
            else:
                bboxes[:, [0, 2]] = w1 - bboxes[:, [2, 0]]

        return blur_image,sharp_image,bboxes

    def random_crop(self, blur_image, sharp_image, bboxes):
        if random.random() < 0.5:
            h1, w1, _ = blur_image.shape
            h2, w2, _ = sharp_image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            if h1==h2 and w1==w2:
                max_r_trans = w1 - max_bbox[2]
                max_d_trans = h1 - max_bbox[3]
            else:
                 print("images do not match...")                

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w1, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h1, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            blur_image = blur_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            sharp_image = sharp_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return blur_image,sharp_image, bboxes

    def random_translate(self, blur_image, sharp_image, bboxes):
        if random.random() < 0.5:
            h1, w1, _ = blur_image.shape
            h2, w2, _ = sharp_image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            if h1==h2 and w1==w2:
                max_r_trans = w1 - max_bbox[2]
                max_d_trans = h1 - max_bbox[3]
            else:
                 print("images do not match...")

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            blur_image = cv2.warpAffine(blur_image, M, (w1, h1))
            sharp_image = cv2.warpAffine(sharp_image, M, (w1, h1))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return blur_image,sharp_image, bboxes

    def str_to_int(self,x):
        return int(float(x))

    def parse_annotation(self, annotation): 
        line_list = annotation.split()       
        blur_image_path = line_list[0]
        sharp_image_path = line_list[0].replace('VOCdevkit','sharp')
        
        if not os.path.exists(blur_image_path):
            raise KeyError("%s does not exist ... " % blur_image_path)
        if not os.path.exists(sharp_image_path):
            raise KeyError("%s does not exist ... " % sharp_image_path)
        blur_image = cv2.imread(blur_image_path)
        sharp_image = cv2.imread(sharp_image_path)
        try:
            blur_image.shape[2] != 3
        except:
            print("{} not exist...".format(blur_image_path))

        try:
            sharp_image.shape[2] != 3
        except:
            print("{} not exist...".format(sharp_image_path))

        if self.dataset_type == "converted_coco":
            boxes = []
            num_of_boxes = (len(line_list) - 3) / 5
            if int(num_of_boxes) == num_of_boxes:
                num_of_boxes = int(num_of_boxes)
            else:
                raise ValueError("num_of_boxes must be 'int'.")
            for index in range(num_of_boxes):                
                xmin = self.str_to_int(line_list[3 + index * 5])/4
                ymin = self.str_to_int(line_list[3 + index * 5 + 1])/4
                xmax = self.str_to_int(line_list[3 + index * 5 + 2])/4
                ymax = self.str_to_int(line_list[3 + index * 5 + 3])/4               
                class_id = int(line_list[3 + index * 5 + 4])
                boxes.append([int(xmin),int(ymin),int(xmax),int(ymax),class_id])
            bboxes = np.array(boxes)

        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = np.array(
                [list(map(float, box.split(","))) for box in line[1:]]
            )
            bboxes = bboxes * np.array([width, height, width, height, 1])
            bboxes = bboxes.astype(np.int64)

        if self.data_aug:              
            blur_image, sharp_image, bboxes = self.random_horizontal_flip(
                np.copy(blur_image), np.copy(sharp_image), np.copy(bboxes)
            )
            blur_image, sharp_image, bboxes = self.random_crop(np.copy(blur_image), np.copy(sharp_image), np.copy(bboxes))
            blur_image, sharp_image, bboxes = self.random_translate(
                np.copy(blur_image), np.copy(sharp_image), np.copy(bboxes)
            )
        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
        sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)

        blur_image, sharp_image, bboxes = utils.image_preprocess(   
            np.copy(blur_image),
            np.copy(sharp_image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )   
        return blur_image, sharp_image, bboxes


    def preprocess_true_boxes(self, bboxes):       
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(self.N_planes)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(self.N_planes)]#(1,150,4)
        bbox_count = np.zeros((self.N_planes,))

        for bbox in bboxes:  
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            
##            onehot = np.zeros(self.num_classes, dtype=np.float)
##            onehot[bbox_class_ind] = 1.0
##            uniform_distribution = np.full(
##                self.num_classes, 1.0 / self.num_classes
##            )
##            deta = 0.01
##            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )               

            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]   
            )

            iou = []
            exist_positive = False
            for i in range(self.N_planes):       
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))     
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5 
                )

                anchors_xywh[:, 2:4] = self.anchors[i]/self.strides[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh 
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.2

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh   
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
##                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    label[i][yind, xind, iou_mask, 5:] = 1.0

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh    
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:   
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
##                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
                label[best_detect][yind, xind, best_anchor, 5:] = 0.995

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label  
        sbboxes, mbboxes, lbboxes  = bboxes_xywh   
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes  

    def __len__(self):
        return self.num_batchs


