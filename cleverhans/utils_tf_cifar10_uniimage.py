
from cleverhans.utils_tf import *
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
import logging

import logging
import os
import time
import warnings

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans import canary
from cleverhans.train import avg_grads
from cleverhans.utils import safe_zip
from matplotlib import pyplot as plt

_logger = create_logger("cleverhans.utils.tf")
_logger.setLevel(logging.INFO)


def train_with_noise(sess, loss, x_train, y_train,
          init_all=False, evaluate=None, feed=None, args=None,
          rng=None, var_list=None, fprop_args=None, optimizer=None,
          devices=None, x_batch_preprocessor=None, use_ema=False,
          ema_decay=.998, run_canary=None,
          loss_threshold=1e5, dataset_train=None, dataset_size=None,
          save=False):
  """
  Run (optionally multi-replica, synchronous) training to minimize `loss`
  :param sess: TF session to use when training the graph
  :param loss: tensor, the loss to minimize
  :param x_train: numpy array with training inputs or tf Dataset
  :param y_train: numpy array with training outputs or tf Dataset
  :param init_all: (boolean) If set to true, all TF variables in the session
                   are (re)initialized, otherwise only previously
                   uninitialized variables are initialized before training.
  :param evaluate: function that is run after each training iteration
                   (typically to display the test/validation accuracy).
  :param feed: An optional dictionary that is appended to the feeding
               dictionary before the session runs. Can be used to feed
               the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `nb_epochs`, `learning_rate`,
               `batch_size`
  :param rng: Instance of numpy.random.RandomState
  :param var_list: Optional list of parameters to train.
  :param fprop_args: dict, extra arguments to pass to fprop (loss and model).
  :param optimizer: Optimizer to be used for training
  :param devices: list of device names to use for training
      If None, defaults to: all GPUs, if GPUs are available
                            all devices, if no GPUs are available
  :param x_batch_preprocessor: callable
      Takes a single tensor containing an x_train batch as input
      Returns a single tensor containing an x_train batch as output
      Called to preprocess the data before passing the data to the Loss
  :param use_ema: bool
      If true, uses an exponential moving average of the model parameters
  :param ema_decay: float or callable
      The decay parameter for EMA, if EMA is used
      If a callable rather than a float, this is a callable that takes
      the epoch and batch as arguments and returns the ema_decay for
      the current batch.
  :param loss_threshold: float
      Raise an exception if the loss exceeds this value.
      This is intended to rapidly detect numerical problems.
      Sometimes the loss may legitimately be higher than this value. In
      such cases, raise the value. If needed it can be np.inf.
  :param dataset_train: tf Dataset instance.
      Used as a replacement for x_train, y_train for faster performance.
    :param dataset_size: integer, the size of the dataset_train.
  :return: True if model trained
  """

  # Check whether the hardware is working correctly
  canary.run_canary()
  if run_canary is not None:
    warnings.warn("The `run_canary` argument is deprecated. The canary "
                  "is now much cheaper and thus runs all the time. The "
                  "canary now uses its own loss function so it is not "
                  "necessary to turn off the canary when training with "
                  " a stochastic loss. Simply quit passing `run_canary`."
                  "Passing `run_canary` may become an error on or after "
                  "2019-10-16.")

  args = _ArgsWrapper(args or {})
  fprop_args = fprop_args or {}

  # Check that necessary arguments were given (see doc above)
  # Be sure to support 0 epochs for debugging purposes
  if args.nb_epochs is None:
    raise ValueError("`args` must specify number of epochs")
  if optimizer is None:
    if args.learning_rate is None:
      raise ValueError("Learning rate was not given in args dict")
  assert args.batch_size, "Batch size was not given in args dict"

  if rng is None:
    rng = np.random.RandomState()

  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  else:
    if not isinstance(optimizer, tf.train.Optimizer):
      raise ValueError("optimizer object must be from a child class of "
                       "tf.train.Optimizer")

  grads = []
  xs = []
  preprocessed_xs = []
  ys = []
  if dataset_train is not None:
    assert x_train is None and y_train is None and x_batch_preprocessor is None
    if dataset_size is None:
      raise ValueError("You must provide a dataset size")
    data_iterator = dataset_train.make_one_shot_iterator().get_next()
    x_train, y_train = sess.run(data_iterator)

  devices = infer_devices(devices)
  for device in devices:
    with tf.device(device):
      x = tf.placeholder(x_train.dtype, (None,) + x_train.shape[1:])
      y = tf.placeholder(y_train.dtype, (None,) + y_train.shape[1:])
      xs.append(x)
      ys.append(y)

      if x_batch_preprocessor is not None:
        x = x_batch_preprocessor(x)

      # We need to keep track of these so that the canary can feed
      # preprocessed values. If the canary had to feed raw values,
      # stochastic preprocessing could make the canary fail.
      preprocessed_xs.append(x)

      loss_value = loss.fprop(x, y, **fprop_args)

      grads.append(optimizer.compute_gradients(
          loss_value, var_list=var_list))
  num_devices = len(devices)
  print("num_devices: ", num_devices)

  grad = avg_grads(grads)
  # Trigger update operations within the default graph (such as batch_norm).
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.apply_gradients(grad)

  epoch_tf = tf.placeholder(tf.int32, [])
  batch_tf = tf.placeholder(tf.int32, [])

  if use_ema:
    if callable(ema_decay):
      ema_decay = ema_decay(epoch_tf, batch_tf)
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    with tf.control_dependencies([train_step]):
      train_step = ema.apply(var_list)
    # Get pointers to the EMA's running average variables
    avg_params = [ema.average(param) for param in var_list]
    # Make temporary buffers used for swapping the live and running average
    # parameters
    tmp_params = [tf.Variable(param, trainable=False)
                  for param in var_list]
    # Define the swapping operation
    param_to_tmp = [tf.assign(tmp, param)
                    for tmp, param in safe_zip(tmp_params, var_list)]
    with tf.control_dependencies(param_to_tmp):
      avg_to_param = [tf.assign(param, avg)
                      for param, avg in safe_zip(var_list, avg_params)]
    with tf.control_dependencies(avg_to_param):
      tmp_to_avg = [tf.assign(avg, tmp)
                    for avg, tmp in safe_zip(avg_params, tmp_params)]
    swap = tmp_to_avg

  batch_size = args.batch_size

  assert batch_size % num_devices == 0
  device_batch_size = batch_size // num_devices

  saver = tf.train.Saver(max_to_keep=100)
  if init_all:
    sess.run(tf.global_variables_initializer())
  else:
    initialize_uninitialized_global_variables(sess)

  for epoch in xrange(args.nb_epochs):

    feed_x_train = convert_uniimage(np.clip(x_train+(np.random.uniform(0, 0.8, (len(x_train), 32, 32, 3)) - 0.4), 0, 1))
    if dataset_train is not None:
      nb_batches = int(math.ceil(float(dataset_size) / batch_size))
    else:
      # Indices to shuffle training set
      index_shuf = list(range(len(x_train)))
      # Randomly repeat a few training examples each epoch to avoid
      # having a too-small batch
      while len(index_shuf) % batch_size != 0:
        index_shuf.append(rng.randint(len(x_train)))
      nb_batches = len(index_shuf) // batch_size
      rng.shuffle(index_shuf)
      # Shuffling here versus inside the loop doesn't seem to affect
      # timing very much, but shuffling here makes the code slightly
      # easier to read
      x_train_shuffled = feed_x_train[index_shuf]
      y_train_shuffled = y_train[index_shuf]

    prev = time.time()
    for batch in range(nb_batches):
      if dataset_train is not None:
        x_train_shuffled, y_train_shuffled = sess.run(data_iterator)
        start, end = 0, batch_size
      else:
        # Compute batch start and end indices
        start = batch * batch_size
        end = (batch + 1) * batch_size
        # Perform one training step
        diff = end - start
        assert diff == batch_size

      feed_dict = {epoch_tf: epoch, batch_tf: batch}
      for dev_idx in xrange(num_devices):
        cur_start = start + dev_idx * device_batch_size
        cur_end = start + (dev_idx + 1) * device_batch_size
        feed_dict[xs[dev_idx]] = x_train_shuffled[cur_start:cur_end]
        feed_dict[ys[dev_idx]] = y_train_shuffled[cur_start:cur_end]
      if cur_end != end and dataset_train is None:
        msg = ("batch_size (%d) must be a multiple of num_devices "
               "(%d).\nCUDA_VISIBLE_DEVICES: %s"
               "\ndevices: %s")
        args = (batch_size, num_devices,
                os.environ['CUDA_VISIBLE_DEVICES'],
                str(devices))
        raise ValueError(msg % args)
      if feed is not None:
        feed_dict.update(feed)

      _, loss_numpy = sess.run(
          [train_step, loss_value], feed_dict=feed_dict)

      if np.abs(loss_numpy) > loss_threshold:
        raise ValueError("Extreme loss during training: ", loss_numpy)
      if np.isnan(loss_numpy) or np.isinf(loss_numpy):
        raise ValueError("NaN/Inf loss during training")
    assert (dataset_train is not None or
            end == len(index_shuf))  # Check that all examples were used
    cur = time.time()
    _logger.info("Epoch " + str(epoch) + " took " +
                 str(cur - prev) + " seconds")
    if evaluate is not None:
      if use_ema:
        # Before running evaluation, load the running average
        # parameters into the live slot, so we can see how well
        # the EMA parameters are performing
        sess.run(swap)
      evaluate()
      if use_ema:
        # Swap the parameters back, so that we continue training
        # on the live parameters
        sess.run(swap)

    if save and ((epoch + 1) % 50 == 0 or (epoch + 1) == args.nb_epochs):
      with tf.device('/CPU:0'):
        save_path = os.path.join(args.train_dir, args.filename)
        if tf.gfile.Exists(args.train_dir) == False:
          tf.gfile.MakeDirs(args.train_dir)
        saver.save(sess, save_path, global_step=(epoch + 1))
      _logger.info("Reaching save point at " + str(epoch + 1) + ": " +
                   str(save_path))

  if use_ema:
    # When training is done, swap the running average parameters into
    # the live slot, so that we use them when we deploy the model
    sess.run(swap)



  return True


def model_eval(sess, x, y, predictions, X_test=None, Y_test=None,
               feed=None, args=None, is_adv=None, ae=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `batch_size`
  :return: a float with the accuracy value
  """
  global _model_eval_cache
  args = _ArgsWrapper(args or {})

  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument "
                     "must be supplied.")

  # Define accuracy symbolically
  key = (y, predictions)
  if key in _model_eval_cache:
    correct_preds = _model_eval_cache[key]
  else:
    correct_preds = tf.equal(tf.argmax(y, axis=-1),
                             tf.argmax(predictions, axis=-1))
    _model_eval_cache[key] = correct_preds

  # Init result var
  accuracy = 0.0
  percent_perturbed = 0.0
  uniImagePreds = []

  # X_test = convert_uniimage(X_test)
  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
    for batch in range(nb_batches):
      if batch % 100 == 0 and batch > 0:
        _logger.debug("Batch " + str(batch))

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * args.batch_size
      end = min(len(X_test), start + args.batch_size)

      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      Y_cur[:cur_batch_size] = Y_test[start:end]
      xtest = np.copy(X_cur)

      ###############################
      # Generate adversarial images #
      ###############################
      adv_x = np.copy(1-X_cur)
      # print(is_adv, ae)
      if is_adv and ae != None:
        # print(X_cur.shape)
        feed_dict = {x: X_cur}
        adv_x = ae.eval(feed_dict=feed_dict)

      ###############################
      # Transform image to uniimage #
      ###############################
      # tmpX = np.clip(adv_x+(np.random.uniform(0, 0.8, (len(adv_x), 32, 32, 3)) - 0.4), 0, 1)
      # X_cur = convert_uniimage(tmpX)
      # X_cur = adv_x
      # X_cur = convert_uniimage(adv_x)
      # X_cur = convert_uniimage(tmpX)
        # print("adv_x", adv_x.shape)
        # X_cur = adv_x
        # print(X_cur.shape)




      ##################
      # Showing images #
      ##################
      showImg = True
      showImg = False
      if showImg:
        imgIndex = 8
        for iii in range(len(X_cur)):
          fig = plt.figure()
          pixels = xtest[iii].reshape((32, 32, 3))
          sub = fig.add_subplot(1, 4, 1)
          plt.imshow(pixels, cmap='gray')
          pixels = adv_x[iii].reshape((32, 32, 3))
          sub = fig.add_subplot(1, 4, 2)
          plt.imshow(pixels, cmap='gray')
          pixels = tmpX[iii].reshape((32, 32, 3))
          sub = fig.add_subplot(1, 4, 3)
          plt.imshow(pixels, cmap='gray')
          pixels = X_cur[iii].reshape((32, 32, 3))
          sub = fig.add_subplot(1, 4, 4)
          plt.imshow(pixels, cmap='gray')
          # pixels = adv_x[iii].reshape((28, 28)) - xtrain[iii].reshape((28, 28))
          # print(np.mean(np.sum((adv_x[iii:iii+1] - xtrain[iii:iii+1]) ** 2,
          #        axis=(1, 2, 3)) ** .5))
          # sub = fig.add_subplot(1, 3, iii+3)
          # plt.imshow(pixels / abs(pixels).max() * 0.2 + 0.5, cmap='gray')

          ##########################
          # The significant changes of image from CW attack
          # 1st loop 3rd picture
          # 3rd loop 5th picture
          # 4th loop 7th picture
          # 10th loop 3rd picture
          ##########################
          # path_ori = "/home/labiai/Jiacang/Experiments/Images/example4/"
          # ## saving original images
          # k = 1000
          # output_original = path_ori + "original/" + str((batch * nb_batches) + k) + ".png"
          # print("writing " + output_original)
          # with open(output_original, "wb") as g:
          #   w = png.Writer(28, 28, greyscale=True)
          #   x_res = X_test[imgIndex]
          #   w.write(g, x_res.reshape(28, 28) * 255)
          # output_original = path_ori + "cw/" + str((batch * nb_batches) + k) + ".png"
          # with open(output_original, "wb") as g:
          #   w = png.Writer(28, 28, greyscale=True)
          #   x_res = adv_x[imgIndex]
          #   w.write(g, x_res.reshape(28, 28) * 255)
          # output_original = path_ori + "difference/" + str((batch * nb_batches) + k) + ".png"
          # with open(output_original, "wb") as g:
          #   w = png.Writer(28, 28, greyscale=True)
          #   x_res = adv_x[imgIndex].reshape((28, 28)) - xtrain[imgIndex].reshape((28, 28))
          #   x_res = x_res / abs(x_res).max() * 0.2 + 0.5
          #   w.write(g, x_res.reshape(28, 28) * 255)
          # if iii == 0:
          #   break
          plt.show()


      feed_dict = {x: X_cur, y: Y_cur}
      # print(feed_dict)
      if feed is not None:
        feed_dict.update(feed)
      cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

      accuracy += cur_corr_preds[:cur_batch_size].sum()

      # Compute the percentage of perturbation if they are adversarial images
      if is_adv and ae != None:
        percent_perturbed += np.mean(np.sum((adv_x - xtest) ** 2,
                       axis=(1, 2, 3)) ** .5) / nb_batches

    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)

  return accuracy

_model_eval_cache = {}
