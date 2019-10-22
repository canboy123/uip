"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
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
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
from cleverhans.compat import flags
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.utils_tf_uniimage import model_eval, tf_model_load, convert_uniimage, transform_4_in_1, tf_transform_4_in_1, tf_avg_4_in_1, train_with_noise
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod, HopSkipJumpAttack, BasicIterativeMethod, MomentumIterativeMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN, AlexNet

FLAGS = flags.FLAGS

NB_EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

discretizeColor = 32
retrain = False
save = True
folder = "uniimage"
folder = "discretize"
type = 'normal'
type = 'noise'
# type = 'non-uniimage'
file = "/home/tako/Jiacang/Experiments/tmp/data/mnist"
save_dir = '/home/tako/Jiacang/Experiments/tmp/'+folder+'/'+type+'/mnist_'+str(1000)+'/'
eps = 0.3
file = "/home/tako/Jiacang/Experiments/tmp/data/fashion_mnist"
save_dir = '/home/tako/Jiacang/Experiments/tmp/'+folder+'/'+type+'/fmnist_'+str(1000)+'/'
eps = 8/256

print("file:", file)
filename = 'network'

def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
                   clean_train=CLEAN_TRAIN,
                   testing=False,
                   backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                   nb_filters=NB_FILTERS, num_threads=None,
                   label_smoothing=0.1):
  """
  MNIST cleverhans tutorial
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

  # Get MNIST data
  mnist = MNIST(path=file, train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # x_train = x_train[0:1].reshape(784)
  # k = np.unique(x_train.reshape(-1, 784))
  # k = list(set(x_train.reshape(784)))
  # nk = [k.index(x_train[x]) for x in len(x_train)]
  # print(k, np.shape(k), nk)

  ###############################
  # Transform image to uniimage #
  ###############################
  # x_train = convert_uniimage(x_train)
  # x_test = transform_4_in_1(x_test)
  # trans_x_text = np.copy(x_test)
  # x_test = convert_uniimage(x_test)
  # uni_x_test = np.copy(x_test)


  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

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
      'eps': eps,
      'clip_min': 0.,
      'clip_max': 1.
  }
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(preds, x_set, y_set, report_key, is_adv=None, ae=None, type=None, datasetName=None, discretizeColor=1):
    accuracy, distortion = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params, is_adv=is_adv, ae=ae,
                                      type=type, datasetName=datasetName, discretizeColor=discretizeColor)
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
    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)

    def evaluate():
      do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False, type=type,
                                     datasetName="MNIST", discretizeColor=discretizeColor)

    saveFileNumArr = []
    # saveFileNumArr = [50, 500, 1000]

    count = 0
    appendNum = 50
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
        # train(sess, loss, x_train, y_train, evaluate=evaluate,
        #       args=train_params, rng=rng, var_list=model.get_params())
        train_with_noise(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params, rng=rng,
                         var_list=model.get_params(), save=save, type=type, datasetName="MNIST", retrain=retrain,
                         discretizeColor=discretizeColor)

      # Calculate training error
      accuracy, distortion = do_eval(preds, x_test, y_test, 'train_clean_train_clean_eval', False, type=type,
                                     datasetName="MNIST", discretizeColor=discretizeColor)

      # Initialize the Fast Gradient Sign Method (FGSM) attack object and
      # graph
      fgsm = FastGradientMethod(model, sess=sess)
      # fgsm = BasicIterativeMethod(model, sess=sess)
      # fgsm = MomentumIterativeMethod(model, sess=sess)
      # fgsm_params = {
      #   'clip_min': 0.,
      #   'clip_max': 1.,
      #   'verbose': False
      # }
      # fgsm = HopSkipJumpAttack(model, sess=sess)
      adv_x = fgsm.generate(x, **fgsm_params)
      # adv_x = fgsm.generate_np(x, **fgsm_params)
      # adv = sess.run(adv_x, feed_dict={x: x_test})
      preds_adv = model.get_logits(adv_x)
      # print(sess.run(preds_adv, feed_dict={x: x_test}))

      #############################
      # Create adversarial images #
      #############################
      # We have to produce adversarial image 1 by 1 by using HopSkipJumpAttack
      # adv_test = []
      # for i in range(len(x_test)):
      #   tmp_adv_test = sess.run(adv_x, feed_dict={x: [x_test[i]]})
      #   adv_test.append(tmp_adv_test[0])
      #   if (i+1) % 100 == 0:
      #     print((i+1),"/",len(x_test), " adversarial images")
      #
      # adv_test = np.array(adv_test)
      # print(np.shape(adv_test))



      # Evaluate the accuracy of the MNIST model on adversarial examples
      # do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)
      # accuracy, distortion = do_eval(preds, x_test, y_test, 'clean_train_adv_eval', True, ae=adv_x,
      #                                type=type, datasetName="MNIST", discretizeColor=discretizeColor)
      # do_eval(preds, adv_test, y_test, 'clean_train_adv_eval', True)

      distortionArr.append(distortion)
      accuracyArr.append(accuracy)
      # print(str(accuracy))
      # print(str(distortion))

    print("accuracy:")
    for accuracy in accuracyArr:
        print(accuracy)

    print("distortion:")
    for distortion in distortionArr:
        print(distortion)

  return report


def main(argv=None):
  """
  Run the tutorial using command line flags.
  """
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 clean_train=FLAGS.clean_train,
                 backprop_through_attack=FLAGS.backprop_through_attack,
                 nb_filters=FLAGS.nb_filters)


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
