import tensorflow as tf
import copy
from core.image_pool import ImagePool          
from core.config import cfg
from core import utils

class PerceptualLoss():

    def contentFunc(self):
        vgg = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights="imagenet",input_shape=(cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3)
        )
        model = tf.keras.models.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
        )
        model.trainable = False  
        return model

    def initialize(self):   
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
 
        f_fake = self.contentFunc(fakeIm, training=False)
        f_real = self.contentFunc(realIm, training=False)
        f_real_no_grad = tf.stop_gradient(f_real)       
        Perceptual_loss = tf.keras.losses.mean_squared_error(f_fake, f_real_no_grad) 
        loss_content = tf.keras.losses.mean_squared_error(fakeIm, realIm)
        print("Perceptual_loss:",0.005 * tf.reduce_mean(Perceptual_loss))
        print("Content_loss:",0.5*tf.reduce_mean(loss_content))
        return 0.005 * tf.reduce_mean(Perceptual_loss) + 0.5*tf.reduce_mean(loss_content)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


class RelativisticDiscLossLS():          

    def __init__(self):
        self.fake_pool = ImagePool(50)  
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        self.pred_fake = net(fakeB, training=False)
        self.pred_real = net(realB, training=False)
        
        errG = (tf.reduce_mean((self.pred_real - tf.reduce_mean(self.fake_pool.query()) + 1) ** 2) +
                tf.reduce_mean((self.pred_fake - tf.reduce_mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        self.fake_B = tf.stop_gradient(fakeB)
        self.real_B = realB 

        self.pred_fake = net(self.fake_B,training=False) 
        self.fake_pool.add(self.pred_fake)

        self.pred_real = net(realB,training=False)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (tf.reduce_mean((self.pred_real - tf.reduce_mean(self.fake_pool.query()) - 1) ** 2) +
                       tf.reduce_mean((self.pred_fake - tf.reduce_mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)

def get_loss():
    content_loss = PerceptualLoss()
    content_loss.initialize()
    
    disc_loss = RelativisticDiscLossLS()
    return content_loss, disc_loss

class DoubleGAN(object):
    def __init__(self, patch_model,full_model, criterion):
        self.patch_d = patch_model
        self.full_d = full_model
        self.full_criterion = copy.deepcopy(criterion)
        self.criterion = criterion

    def loss_d(self, pred, gt):
        return (self.criterion(self.patch_d, pred, gt) + self.full_criterion(self.full_d, pred, gt)) / 2

    def loss_g(self, pred, gt):
        return (self.criterion.get_g_loss(self.patch_d, pred, gt) + self.full_criterion.get_g_loss(self.full_d, pred,
                                                                                                  gt)) / 2
    def get_params(self):
        return list(self.patch_d.trainable_variables) + list(self.full_d.trainable_variables)


class SingleGAN(object):
    def __init__(self, net_d, criterion):
        self.net_d = net_d
        self.criterion = criterion
        
    def loss_d(self, pred, gt):
        return self.criterion(self.net_d, pred, gt)

    def loss_g(self, pred, gt):
        return self.criterion.get_g_loss(self.net_d, pred, gt)

    def get_params(self):
        return self.net_d.trainable_variables




