import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from core.yolov5 import NLayerDiscriminator, YOLO, decode, compute_loss, decode_train,filter_boxes
from core.dataset import Dataset
from core.config import cfg, CFLAGS
from core import utils
from core.utils import freeze_all, unfreeze_all
from core.deblur_losses import get_loss, DoubleGAN, SingleGAN
FLAGS = CFLAGS()

    
def main():
    INPUT_SIZE = FLAGS.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)           
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    predicted_dir_path = './AP/predicted'
    loss_dir_path = './AP/loss'
    text_result_path = './AP/detect'    

    trainset = Dataset(FLAGS, is_training=True)
    adv_lambda = 0.001         
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch 

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    feature_maps = YOLO(FLAGS.scale_v5, input_layer, NUM_CLASS) 
    
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        elif i == 1:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        elif i==2:
            bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        else:
            result_G = fm
    bbox_tensors.append(result_G)

    D2_model = tf.keras.Model(input_layer, bbox_tensors)
    full_model = NLayerDiscriminator(ndf=64, n_layers=5)  
    full_model.build([None, cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])

    optimizer_D2 = tf.keras.optimizers.Adam()
    optimizer_Dis = tf.keras.optimizers.SGD()
    criterionG, criterionD = get_loss()     

    if cfg.TRAIN.DoubleGAN:
        patch_model = NLayerDiscriminator(ndf=64, n_layers=3)
        patch_model.build([None, cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
        adv_trainer = DoubleGAN(patch_model, full_model, criterionD)
    else:
        adv_trainer = SingleGAN(full_model, criterionD)

    if cfg.TRAIN.GRADNORM:
        optimizer_P = tf.keras.optimizers.SGD()
        
    D2_model.summary()
    T_freeze_layers = utils.load_True_freeze_layer(FLAGS.scale_v5)

    # metrics draw now
    total_loss_metric = tf.metrics.Mean()
    loss1_metric = tf.metrics.Mean()
    loss2_metric = tf.metrics.Mean()
    total_loss_result= []
    loss1_result= []
    loss2_result= []
    Weightloss1 = tf.Variable(1.0)
    Weightloss2 = tf.Variable(1.0)
    params = [Weightloss1, Weightloss2]
    alph = 0.16

##    @tf.function
    def train_step(image_data, target):
        start_time = time.time()        
        # with tf.GradientTape() as tape1,tf.GradientTape() as tape2:
        ##Experiments have found that this performance is better than "persistent=True".
        with tf.GradientTape() as tape1,tf.GradientTape() as tape2,tf.GradientTape() as tape3,tf.GradientTape() as tape4,tf.GradientTape() as tape5:
            pred_result = D2_model(image_data[0], training=True)
            G_im = pred_result[-1]
            loss_D = loss_content = loss_adv = loss_G = giou_loss = conf_loss = prob_loss = 0
            
            #update Discriminator
            loss_D = 1000 * adv_lambda * adv_trainer.loss_d(G_im, image_data[1])
            gradients_Dis = tape1.gradient(loss_D, adv_trainer.get_params())
            optimizer_Dis.apply_gradients(zip(gradients_Dis, adv_trainer.get_params()))

            #update D2Net    
            loss_content = criterionG(G_im, image_data[1])
            loss_adv = adv_trainer.loss_g(G_im, image_data[1])  
            loss_G = 1000*(loss_content + adv_lambda * loss_adv)

            for i in range(ANCHORS.shape[0]):     
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            yolo_loss = giou_loss + conf_loss + prob_loss

            l1 = params[0]*yolo_loss
            l2 = params[1]*loss_G
            total_loss = (l1 + l2)/2
            gradients_D2 = tape2.gradient(total_loss, D2_model.trainable_variables)
            optimizer_D2.apply_gradients(zip(gradients_D2, D2_model.trainable_variables))

            ###Gradnorm###
            L0 = 183            
            LP = D2_model.trainable_variables[162]   #D_conv2d_54
            G1R = tape3.gradient(l1, LP)
            G1 = tf.norm(G1R, ord=2)
            G2R = tape4.gradient(l2, LP)
            G2 = tf.norm(G2R, ord=2)
            G_avg = (G1+G2)/2
            # Calculating relative losses 
            lhat1 = (l1)/L0
            lhat2 = (l2)/L0
            lhat_avg = (lhat1 + lhat2)/2            
            inv_rate1 = lhat1/lhat_avg
            inv_rate2 = lhat2/lhat_avg
            C1 = G_avg*(inv_rate1)**alph
            C2 = G_avg*(inv_rate2)**alph
            C1 = tf.stop_gradient(tf.identity(C1))
            C2 = tf.stop_gradient(tf.identity(C2))

            # Gradnorm loss 
            loss_gradnorm = tf.math.reduce_sum(tf.math.abs(G1-C1)) + tf.math.reduce_sum(tf.math.abs(G2-C2))
            grad_grad = tape5.gradient(loss_gradnorm, params)
            optimizer_P.apply_gradients(grads_and_vars=zip(grad_grad, params)) 

        total_loss_metric.update_state(values=total_loss)
        loss1_metric.update_state(values=yolo_loss)
        loss2_metric.update_state(values=loss_G)
        time_per_step = time.time() - start_time
        print("Step: {}/{},lr: {:.6f}, {:.2f}s/step, total_loss: {:.5f}, "
              "yolo loss: {:.5f}, G_loss: {:.5f}, loss_adv: {:.5f}, D_loss: {:.5f}".format(
                                                          global_steps.numpy(),
                                                          total_steps,
                                                          optimizer_D2.lr.numpy(),
                                                          time_per_step,
                                                          total_loss,
                                                          yolo_loss,
                                                          loss_G,
                                                          adv_lambda*loss_adv,
                                                          loss_D
                                                          ))
        
        # update learning rate
        global_steps.assign_add(1)      
        if global_steps < warmup_steps:  
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT  
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer_D2.lr.assign(lr.numpy())  
        optimizer_Dis.lr.assign(10*lr.numpy()) 

        loss_list_step = [optimizer_D2.lr.numpy(),total_loss,yolo_loss,
                    loss_G,loss_D,giou_loss,conf_loss, prob_loss,loss_content,adv_lambda * loss_adv]
        return np.array(loss_list_step)


##    @tf.function
    def test_epoch(D2_model,dectect_epoch_path):        
        with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:                         
            for num, line in enumerate(annotation_file):                
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                predict_result_path = os.path.join(predicted_epoch_path, str(image_name) + '.txt')

                original_image = cv2.imread(image_path)
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # Predict Process
##                image_letter, ratio, (dw, dh) = utils.letterbox(image)
                image_letter = utils.test_image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
                image_data = image_letter[np.newaxis, ...].astype(np.float32)
                batch_data = tf.constant(image_data)            

                bbox_tensors = []
                prob_tensors = []
                pred_result = D2_model(batch_data,training=False)
                G_im = pred_result[-1][0]

                for i in range(ANCHORS.shape[0]):     
                    fm = pred_result[i * 2]
                    if i == 0:
                        output_tensors = decode(fm, FLAGS.size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                    elif i == 1:
                        output_tensors = decode(fm, FLAGS.size // 16, NUM_CLASS, STRIDES, ANCHORS, 1, XYSCALE)
                    elif i==2:
                        output_tensors = decode(fm, FLAGS.size // 32, NUM_CLASS, STRIDES, ANCHORS, 2, XYSCALE)
                    bbox_tensors.append(output_tensors[0])
                    prob_tensors.append(output_tensors[1])
                        
                pred_bbox = tf.concat(bbox_tensors, axis=1)
                pred_prob = tf.concat(prob_tensors, axis=1)
                boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.size, FLAGS.size]))
                pred_bbox = tf.concat([boxes, pred_conf], axis=-1)    

                boxes = pred_bbox[:, :, 0:4]
                pred_conf = pred_bbox[:, :, 4:]
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                        scores=tf.reshape(
                            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                        max_output_size_per_class=1,
                        max_total_size=1,
                        iou_threshold=FLAGS.iou,
                        score_threshold=FLAGS.score
                    )
                boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

                if num % 1 ==0:
                    G_im = pred_result[-1][0]
                    G_im = G_im * 255
                    G_im = np.array(G_im).astype(np.int32)
                    image_result = utils.draw_bbox(np.copy(G_im), [boxes, scores, classes, valid_detections])                    
                    image_result = image_result[:,:,::-1]
                    filepath = dectect_epoch_path+"/"+ str(image_name)
                    cv2.imwrite(filepath, image_result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    ################################################################
    ################################################################
    ################################################################
    if os.path.exists(loss_dir_path): shutil.rmtree(loss_dir_path)    
    os.mkdir(loss_dir_path)
    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch >= first_stage_epochs:
            for name in T_freeze_layers:
                try:
                    freeze = D2_model.get_layer(name)   
                    freeze_all(freeze)
                    print("Successfully freeze {}...".format(name))
                except:
                    print("{} not exist...".format(name)) 

        loss_epoch = np.zeros((steps_per_epoch,10),dtype=np.float32)
        for index, (image_data, target) in enumerate(trainset):
            loss_step = train_step(image_data, target)    
            loss_epoch[index] = loss_step
        mask = loss_epoch[:,0] >0
        loss_mean = np.mean(tf.boolean_mask(loss_epoch,mask),0)
        loss_list_step = {"D2:lr":loss_mean[0],"total_loss":loss_mean[1],"loss/yolo_loss":loss_mean[2],
                        "G_loss":loss_mean[3],"D_loss":loss_mean[4],"loss/giou_loss":loss_mean[5],"loss/conf_loss":loss_mean[6],
                        "loss/prob_loss":loss_mean[7],"loss_content":loss_mean[8],"adv_lambda * loss_adv":loss_mean[9]}
        loss_epoch_path =  os.path.join(loss_dir_path, "epoch-{}".format(epoch) + '.txt')
        with open(loss_epoch_path, 'w') as f:
                for vm in loss_list_step.values():
                    loss_mess = ' '.join([str(vm)]) + '\n'
                    f.write(loss_mess)

        print("No {} epoch params are {} and {}:".format(epoch,params[0].numpy(),params[1].numpy()))
        total_loss_result.append(total_loss_metric.result())
        loss1_result.append(loss1_metric.result())
        loss2_result.append(loss2_metric.result())        
        total_loss_metric.reset_states()
        loss1_metric.reset_states()
        loss2_metric.reset_states()

        if epoch % FLAGS.save_frequency == 0:
            D2_model.save_weights(filepath=FLAGS.save_model_dir+"epoch-{}.h5".format(epoch), save_format="h5")
            full_model.save_weights(filepath=FLAGS.save_model_dir+"Dis"+"epoch-{}".format(epoch), save_format="h5")
            print("No {} epoch saved successfully...".format(epoch))
        
        #Evaluation model
        dectect_epoch_path = text_result_path + "-epoch-{}".format(epoch)
        if os.path.exists(dectect_epoch_path): shutil.rmtree(dectect_epoch_path)
        os.mkdir(dectect_epoch_path)
     
        test_epoch(D2_model,dectect_epoch_path)
        print("Evaluation completed...")

    #####draw#####
    total_loss = np.array(total_loss_result)
    Yolo_loss = np.array(loss1_result)
    G_loss = np.array(loss2_result)
    epochs_range = np.arange(0,epoch+1,1)
    plt.figure(dpi=1000,num=1,figsize=(6, 3))
    plt.plot(epochs_range, total_loss, marker='*',linestyle='-',linewidth=1, markersize=2,label='total_loss')
    plt.plot(epochs_range, Yolo_loss,marker='o', linestyle='-',linewidth=1, markersize=2,label='Yolo_loss')
    plt.plot(epochs_range, G_loss, label='Deblur_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')   
    plt.legend(loc='upper right')
    plt.savefig('Tranin_Loss_result.png',bbox_inches="tight",dpi=1000)
    plt.show()
            
if __name__ == '__main__':
    main()
