"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional, UIPModel
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf_uniimage import model_eval, tf_model_load, train_with_noise, convert_uniimage, color_shift_attack

FLAGS = flags.FLAGS

NB_EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

num_trials = 1000
discretizeColor = 32
retrain = False
save = True
folder = "uniimage"
folder = "discretize"
type = "non-uniimage"
type = "normal"
# type = "noise"
save_dir = '/home/tako/Jiacang/Experiments/tmp/'+folder+'/'+type+'/cifar10_'+str(NB_EPOCHS)+'/'
# save_dir = '/home/tako/Jiacang/Experiments/tmp/'+folder+'/'+type+'/cifar10_'+str(NB_EPOCHS)+'_uipModel/'
filename = 'network'


def cifar10_tutorial(train_start=0, train_end=60000, test_start=0,
                     test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE,
                     clean_train=CLEAN_TRAIN,
                     testing=False,
                     backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                     nb_filters=NB_FILTERS, num_threads=None,
                     label_smoothing=0.1, retrain=False):
  """
  CIFAR10 cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')

  # start = 6
  # end = 10
  # x_test = x_test[start:end]
  # y_test = y_test[start:end]

  ###########################
  # Adjust hue / saturation #
  ###########################
  # hueValue = 0.9
  # saturationValue = 0.9
  # tf_x_test = tf.image.adjust_saturation(tf.image.adjust_hue(x_test, saturationValue), hueValue)
  # tf_x_test = tf.image.adjust_saturation(tx_test, hueValue)
  # x_test = sess.run(tf_x_test)




  ###############################
  # Transform image to uniimage #
  ###############################
  # x_train = convert_uniimage(x_train)

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'train_dir': save_dir,
      'filename': filename,
  }
  eval_params = {'batch_size': batch_size}
  fgsm_params = {
      'eps': 8/255,
      'clip_min': 0.,
      'clip_max': 1.
  }
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(preds, x_set, y_set, report_key, is_adv=None, ae=None, type=None, datasetName=None, discretizeColor=1):
    accuracy, distortion = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params, is_adv=is_adv, ae=ae, type=type,
                                      datasetName=datasetName, discretizeColor=discretizeColor)
    setattr(report, report_key, accuracy)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, accuracy))

    return accuracy, distortion

  if clean_train:
    model = ModelAllConvolutional('model1', nb_classes, nb_filters, input_shape=[32, 32, 3])
    # model = UIPModel('model1', nb_classes, nb_filters, input_shape=[32, 32, 3])
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)

    def evaluate():
      do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False, type=type, datasetName="CIFAR10", discretizeColor=discretizeColor)

    # train(sess, loss, None, None,
    #       dataset_train=dataset_train, dataset_size=dataset_size,
    #       evaluate=evaluate, args=train_params, rng=rng,
    #       var_list=model.get_params(), save=save)


    saveFileNumArr = []
    # saveFileNumArr = [50, 500, 1000]

    count = 0
    appendNum = 1000
    while count < 1000:
      count = count + appendNum
      saveFileNumArr.append(count)

    distortionArr = []
    accuracyArr = []
    for i in range(len(saveFileNumArr)):
      saveFileNum = saveFileNumArr[i]
      model_path = os.path.join(save_dir, filename+"-"+str(saveFileNum))

      print("Trying to load trained model from: "+model_path)
      if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
        print("Load trained model")
      else:
        train_with_noise(sess, loss, x_train, y_train, evaluate=evaluate,
                         args=train_params, rng=rng, var_list=model.get_params(), save=save, type=type,
                         datasetName="CIFAR10", retrain=retrain, discretizeColor=discretizeColor)
        retrain = False

      ##########################################
      # Generate semantic adversarial examples #
      ##########################################
      adv_x, y_test2 = color_shift_attack(sess, x, y, np.copy(x_test), np.copy(y_test), preds, args=eval_params, num_trials=num_trials)
      x_test2 = adv_x
      # convert_uniimage(np.copy(x_test2), np.copy(x_test), discretizeColor)
      accuracy, distortion = do_eval(preds, np.copy(x_test2), np.copy(y_test2), 'clean_train_clean_eval', False,
                                     type=type, datasetName="CIFAR10", discretizeColor=discretizeColor)

      # accuracy, distortion = do_eval(preds, np.copy(x_test), np.copy(y_test), 'clean_train_clean_eval', False, type=type,
      #                                datasetName="CIFAR10", discretizeColor=discretizeColor)

      # # Initialize the Fast Gradient Sign Method (FGSM) attack object and
      # # graph
      # fgsm = FastGradientMethod(model, sess=sess)
      # fgsm = BasicIterativeMethod(model, sess=sess)
      # fgsm = MomentumIterativeMethod(model, sess=sess)
      # adv_x = fgsm.generate(x, **fgsm_params)
      # preds_adv = model.get_logits(adv_x)

      # Evaluate the accuracy of the MNIST model on adversarial examples
      # accuracy, distortion = do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True, type=type)
      # accuracy, distortion = do_eval(preds, x_test, y_test, 'clean_train_adv_eval', True, ae=adv_x, type=type,
      #                                datasetName="CIFAR10", discretizeColor=discretizeColor)

      distortionArr.append(distortion)
      accuracyArr.append(accuracy)
      print(str(accuracy))
      print(str(distortion))

    print("accuracy:")
    for accuracy in accuracyArr:
        print(accuracy)

    print("distortion:")
    for distortion in distortionArr:
        print(distortion)

    # print("hue "+str(hueValue))

  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  cifar10_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters, retrain=retrain)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))

  tf.app.run()
