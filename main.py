from sklearn.model_selection import train_test_split
from train import TrainEngine, SemiTrainEngine, MultiTaskTrainEngine, load_data
from sklearn.decomposition import TruncatedSVD
from modelDesign_1 import bs_masks as masks1
from modelDesign_2 import bs_masks as masks2
from multiprocessing import Process
import tensorflow as tf
import numpy as np
import fire
import os

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def train(data_file,
          label_file,
          save_path,
          pretrained_path=None,
          mask_mode=1,
          learn_svd=False,
          repeat_data_times=1,
          **kwargs):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.threading.set_inter_op_parallelism_threads(4)

    x, y = load_data(data_file, label_file)
    if learn_svd:
        svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    else:
        svd_weight = None

    verbose = kwargs.pop('verbose', 1)
    test_size = kwargs.pop('test_size', 0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x[:len(y)], y / 120, test_size=test_size)

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    train_engine = TrainEngine(save_path,
                               batch_size=kwargs.pop('batch_size', 128),
                               infer_batch_size=kwargs.pop('infer_batch_size', 128),
                               epochs=kwargs.pop('epochs', 100),
                               learning_rate=kwargs.pop('learning_rate', 1e-3),
                               dropout=kwargs.pop('dropout', 0.0),
                               bs_masks=bs_masks,
                               svd_weight=svd_weight,
                               loss_epsilon=kwargs.pop('loss_epsilon', 0.0))

    train_process = Process(target=train_engine,
                            args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid)
                            ),
                            kwargs={
                                'repeat_data_times': repeat_data_times,
                                'pretrained_path': pretrained_path,
                                'verbose': verbose
                            })

    train_process.start()
    train_process.join()


def multi_task_train(data_file,
                     label_file,
                     save_path,
                     pretrained_path=None,
                     mask_mode=1,
                     learn_svd=False,
                     repeat_data_times=1,
                     **kwargs):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.threading.set_inter_op_parallelism_threads(4)

    x, y = load_data(data_file, label_file)
    if learn_svd:
        svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    else:
        svd_weight = None

    verbose = kwargs.pop('verbose', 1)
    test_size = kwargs.pop('test_size', 0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x[:len(y)], y / 120, test_size=test_size)

    x_unlabel = x[len(y):]
    y_unlabel = np.zeros((len(x)-len(y), 2), dtype=np.float32)
    x_train = np.vstack([x_train, x_unlabel])
    y_train = np.vstack([y_train, y_unlabel])

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    train_engine = MultiTaskTrainEngine(save_path,
                                        batch_size=kwargs.pop('batch_size', 128),
                                        infer_batch_size=kwargs.pop('infer_batch_size', 128),
                                        epochs=kwargs.pop('epochs', 100),
                                        learning_rate=kwargs.pop('learning_rate', 1e-3),
                                        dropout=kwargs.pop('dropout', 0.0),
                                        bs_masks=bs_masks,
                                        svd_weight=svd_weight,
                                        loss_epsilon=kwargs.pop('loss_epsilon', 0.0))

    train_process = Process(target=train_engine,
                            args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid)
                            ),
                            kwargs={
                                'repeat_data_times': repeat_data_times,
                                'pretrained_path': pretrained_path,
                                'verbose': verbose
                            })

    train_process.start()
    train_process.join()


def semi_train(data_file,
               label_file,
               save_path,
               pretrained_path=None,
               mask_mode=1,
               learn_svd=False,
               repeat_data_times=1,
               **kwargs):

    tf.config.threading.set_inter_op_parallelism_threads(4)
    x, y = load_data(data_file, label_file)
    if learn_svd:
        svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    else:
        svd_weight = None

    verbose = kwargs.pop('verbose', 1)
    test_size = kwargs.pop('test_size', 0.1)
    unlabel_x = x[len(y):]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x[:len(y)], y / 120, test_size=test_size)

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    train_engine = SemiTrainEngine(save_path,
                                   batch_size=kwargs.pop('batch_size', 128),
                                   infer_batch_size=kwargs.pop('infer_batch_size', 128),
                                   unlabel_x=unlabel_x,
                                   epochs=kwargs.pop('epochs', 100),
                                   learning_rate=kwargs.pop('learning_rate', 1e-3),
                                   dropout=kwargs.pop('dropout', 0.0),
                                   bs_masks=bs_masks,
                                   svd_weight=svd_weight,
                                   loss_epsilon=kwargs.pop('loss_epsilon', 0.0))

    train_process1 = Process(target=train_engine,
                             args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid)
                             ),
                             kwargs={
                                 'repeat_data_times': repeat_data_times,
                                 'pretrained_path': pretrained_path,
                                 'verbose': verbose
                             })
    train_process1.start()
    train_process1.join()

    semi_train_x = np.vstack([x_train, unlabel_x])
    semi_train_y = np.vstack([y_train, train_engine.pred_y])
    pretrain_process1 = Process(target=train_engine,
                                args=(
                                    (semi_train_x, semi_train_y),
                                    (x_valid, y_valid)
                                ),
                                kwargs={
                                    'pretrained_path': save_path,
                                    'verbose': verbose
                                })
    pretrain_process1.start()
    pretrain_process1.join()

    train_process2 = Process(target=train_engine,
                             args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid)
                             ),
                             kwargs={
                                 'repeat_data_times': repeat_data_times,
                                 'pretrained_path': save_path,
                                 'verbose': verbose
                             })
    train_process2.start()
    train_process2.join()


def test(data_file,
         label_file,
         model_path,
         result_file=None,
         mode=1):

    if mode == 1:
        from modelDesign_1 import Model_1 as Model
    elif mode == 2:
        from modelDesign_2 import Model_2 as Model

    x, y = load_data(data_file, label_file)
    model = Model(x.shape[1:], 2, weights_path=model_path)
    pred = model.predict(x)
    rmse = np.mean(np.sqrt(np.sum((y - pred[:len(y)]) ** 2, axis=-1)))
    print('RMSE: ', round(rmse, 4))

    if result_file is not None:
        np.save(result_file, pred.transpose((1, 0)))


if __name__ == '__main__':
    fire.Fire()
