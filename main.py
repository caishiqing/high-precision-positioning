from sklearn.model_selection import train_test_split
from train import TrainEngine, load_data, SemiTrainEngine
from sklearn.decomposition import TruncatedSVD
from modelDesign_1 import bs_masks as masks1
from modelDesign_2 import bs_masks as masks2
from multiprocessing import Process
import tensorflow as tf
import numpy as np
import fire


def train(data_file,
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
                                (x_valid, y_valid),
                                repeat_data_times,
                                pretrained_path,
                                kwargs.pop('verbose', 1)
                            ))

    train_process.start()
    train_process.join()

    # train_engine((x_train, y_train),
    #              (x_valid, y_valid),
    #              repeat_data_times,
    #              pretrained_path,
    #              kwargs.pop('verbose', 1))


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
                                   unlabel_data=unlabel_x,
                                   epochs=kwargs.pop('epochs', 100),
                                   learning_rate=kwargs.pop('learning_rate', 1e-3),
                                   dropout=kwargs.pop('dropout', 0.0),
                                   bs_masks=bs_masks,
                                   svd_weight=svd_weight,
                                   loss_epsilon=kwargs.pop('loss_epsilon', 0.0))

    # train_process = Process(target=train_engine,
    #                         args=(
    #                             (x_train, y_train),
    #                             (x_valid, y_valid),
    #                             repeat_data_times,
    #                             pretrained_path,
    #                             kwargs.pop('verbose', 1)
    #                         ))

    # train_process.start()
    # train_process.join()

    train_engine((x_train, y_train),
                 (x_valid, y_valid),
                 repeat_data_times,
                 pretrained_path,
                 kwargs.pop('verbose', 1))


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
