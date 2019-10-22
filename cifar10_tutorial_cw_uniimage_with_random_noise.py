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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional, UIPModel
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf_uniimage import model_eval, tf_model_load, train_with_noise, convert_uniimage

FLAGS = flags.FLAGS

NB_EPOCHS = 1000
BATCH_SIZE = 128
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
VIZ_ENABLED = False
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 1000
TARGETED = False

discretizeColor = 32
retrain = False
save = True
folder = "uniimage"
folder = "discretize"
type = "non-uniimage"
type = "normal"
type = "noise"
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
                     label_smoothing=0.1, retrain=False,
                      source_samples=SOURCE_SAMPLES,
                      attack_iterations=ATTACK_ITERATIONS,
                      targeted=TARGETED):
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

  ###########################
  # Adjust hue / saturation #
  ###########################
  # hueValue = 0.3
  # tf_x_test = tf.image.adjust_saturation(tf.image.adjust_hue(x_test, hueValue), hueValue)
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






  saveFileNumArr = []
  # saveFileNumArr = [50, 500, 1000]

  count = 0
  while count < 1000:
    count = count + 50
    saveFileNumArr.append(count)

  distortionArr = []
  accuracyArr = []
  for i in range(len(saveFileNumArr)):
    saveFileNum = saveFileNumArr[i]
    model_path = os.path.join(save_dir, filename + "-" + str(saveFileNum))
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 64

    # Define TF model graph
    model = ModelAllConvolutional('model1', nb_classes, nb_filters, input_shape=[32, 32, 3])
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'filename': os.path.split(model_path)[-1]
    }

    rng = np.random.RandomState([2017, 8, 30])

    print("Trying to load trained model from: " + model_path)
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path + ".meta"):
      tf_model_load(sess, model_path)
      print("Load trained model")
    else:
      train(sess, loss, x_train, y_train, args=train_params, rng=rng)
      saver = tf.train.Saver()
      saver.save(sess, model_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    # accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    # assert x_test.shape[0] == test_end - test_start, x_test.shape
    # print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    # report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, sess=sess)

    if targeted:
      adv_inputs = np.array(
          [[instance] * nb_classes for
           instance in x_test[:source_samples]], dtype=np.float32)

      one_hot = np.zeros((nb_classes, nb_classes))
      one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

      adv_inputs = adv_inputs.reshape(
          (source_samples * nb_classes, img_rows, img_cols, nchannels))
      adv_ys = np.array([one_hot] * source_samples,
                        dtype=np.float32).reshape((source_samples *
                                                   nb_classes, nb_classes))
      yname = "y_target"
    else:
      adv_inputs = x_test[:source_samples]
      adv_inputs = x_test

      adv_ys = None
      yname = "y"

    if targeted:
      cw_params_batch_size = source_samples * nb_classes
    else:
      cw_params_batch_size = source_samples
    cw_params = {'binary_search_steps': 1,
                 'max_iterations': attack_iterations,
                 'learning_rate': CW_LEARNING_RATE,
                 'batch_size': cw_params_batch_size,
                 'initial_const': 10}

    adv2 = cw.generate(x, **cw_params)
    cw_params[yname] = adv_ys
    adv_x = None
    # adv_x = cw.generate_np(adv_inputs, **cw_params)

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    if targeted:
      accuracy = model_eval(
          sess, x, y, preds, adv_x, adv_ys, args=eval_params)
    else:
      # err = model_eval(sess, x, y, preds, adv, y_test[:source_samples],
      #                  args=eval_params)
      accuracy, distortion = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params, is_adv=True, ae=adv2,
                                        type=type, datasetName="CIFAR10", discretizeColor=discretizeColor)

    print('--------------------------------------')
    print("load save file: ", saveFileNum)
    # Compute the number of adversarial examples that were successfully found
    # print('Test with adv. examples {0:.4f}'.format(adv_accuracy))
    print('Test accuracy on examples: %0.4f ,distortion: %0.4f' % (accuracy, distortion))

    distortionArr.append(distortion)
    accuracyArr.append(accuracy)
    # print(str(accuracy))
    # print(str(distortion))
    tf.reset_default_graph()

  print("accuracy:")
  for accuracy in accuracyArr:
    print(accuracy)

  print("distortion:")
  for distortion in distortionArr:
    print(distortion)

  # Close TF session
  sess.close()


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
