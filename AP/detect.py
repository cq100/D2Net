import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils

flags.DEFINE_string('weights', 'D2Net.h5', 'path to model file')
flags.DEFINE_string('image', './data/1.jpg', 'path to input image')
flags.DEFINE_string('output', 'result', 'path to output image')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.2, 'score threshold')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    D2_model = tf.keras.models.load_model(FLAGS.weights)
    print('load model from: %s ... ' % FLAGS.weights)

    # Predict Process
    original_image = cv2.imread(FLAGS.image)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #image_letter, ratio, (dw, dh) = utils.letterbox(image)
    image_letter = utils.test_image_preprocess(np.copy(image), [FLAGS.size, FLAGS.size])
    image_data = image_letter[np.newaxis, ...].astype(np.float32)
    batch_data = tf.constant(image_data)

    pred_result = D2_model(batch_data,training=False)
    G_im = pred_result[-1][0]
    boxes = pred_result[0][:, :, 0:4]
    pred_conf = pred_result[0][:, :, 4:]
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

    G_im = pred_result[-1][0]
    G_im = G_im * 255
    G_im = np.array(G_im).astype(np.int32)
    image_result = utils.draw_bbox(np.copy(G_im), [boxes, scores, classes, valid_detections])    
    plt.imshow(image_result)
    plt.show()
    image_result = image_result[:,:,::-1]
    filepath = FLAGS.output + ".jpg"
    cv2.imwrite(filepath, image_result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
