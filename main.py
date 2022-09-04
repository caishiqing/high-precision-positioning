from train import TrainEngine, SemiTrainEngine, MultiTaskTrainEngine, load_data
from sklearn.model_selection import train_test_split, KFold
from modelDesign_1 import ensemble, build_multi_head_bs
from sklearn.decomposition import TruncatedSVD
from modelDesign_1 import bs_masks as masks1
from modelDesign_2 import bs_masks as masks2
from multiprocessing import Process
import tensorflow as tf
import pandas as pd
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
          **kwargs):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
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
                               batch_size=kwargs.get('batch_size', 128),
                               infer_batch_size=kwargs.get('infer_batch_size', 128),
                               epochs=kwargs.get('epochs', 100),
                               steps_per_epoch=kwargs.get('steps_per_epoch'),
                               learning_rate=kwargs.get('learning_rate', 1e-3),
                               dropout=kwargs.get('dropout', 0.0),
                               bs_masks=bs_masks,
                               svd_weight=svd_weight,
                               loss_epsilon=kwargs.get('loss_epsilon', 0.0),
                               verbose=kwargs.pop('verbose', 1))

    train_process = Process(target=train_engine,
                            args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid),
                                pretrained_path
                            ))

    train_process.start()
    train_process.join()

    model = tf.keras.models.load_model(save_path)
    model = build_multi_head_bs(model, bs_masks, 120)
    model.save(save_path)


def train_kfold(data_file,
                label_file,
                save_path,
                kfold=5,
                pretrained_path=None,
                mask_mode=1,
                learn_svd=False,
                **kwargs):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.threading.set_inter_op_parallelism_threads(4)

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    x, y = load_data(data_file, label_file)
    if learn_svd:
        svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    else:
        svd_weight = None

    df = pd.DataFrame()
    df['ids'] = list(range(y))
    df['kfold'] = 0
    kf = KFold(n_splits=kfold, shuffle=True)
    for k, (_, v) in enumerate(kf.split(df)):
        df.loc[v, 'kfold'] = k

    for k in range(kfold):
        train_ids = df[df['kfold'] != k]['ids']
        valid_ids = df[df['kfold'] == k]['ids']
        x_train = x[train_ids]
        y_train = y[train_ids] / 120
        x_valid = x[valid_ids]
        y_valid = y[valid_ids] / 120

        model_path = '{}_{}.h5'.format(save_path.split('.')[0], k)
        train_engine = TrainEngine(model_path,
                                   batch_size=kwargs.get('batch_size', 128),
                                   infer_batch_size=kwargs.get('infer_batch_size', 128),
                                   epochs=kwargs.get('epochs', 100),
                                   steps_per_epoch=kwargs.get('steps_per_epoch'),
                                   learning_rate=kwargs.get('learning_rate', 1e-3),
                                   dropout=kwargs.get('dropout', 0.0),
                                   bs_masks=bs_masks,
                                   svd_weight=svd_weight,
                                   loss_epsilon=kwargs.get('loss_epsilon', 0.0),
                                   verbose=kwargs.pop('verbose', 1))

        train_process = Process(target=train_engine,
                                args=(
                                    (x_train, y_train),
                                    (x_valid, y_valid),
                                    pretrained_path
                                ))

        train_process.start()
        train_process.join()

    models = []
    for k in range(kfold):
        model_path = '{}_{}.h5'.format(save_path.split('.')[0], k)
        model = tf.keras.models.load_model(model_path, compile=False)
        model = build_multi_head_bs(model, bs_masks, 120)
        models.append(model)
        os.remove(model_path)

    model = ensemble(models)
    model.save(save_path)


def multi_task_train(data_file,
                     label_file,
                     save_path,
                     pretrained_path=None,
                     mask_mode=1,
                     learn_svd=False,
                     **kwargs):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.threading.set_inter_op_parallelism_threads(4)

    x, y = load_data(data_file, label_file)
    if learn_svd:
        svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    else:
        svd_weight = None

    test_size = kwargs.pop('test_size', 0.1)
    y = np.vstack([y, np.zeros((len(x)-len(y), 2))]) * 0
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y / 120, test_size=test_size)

    del x, y

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    train_engine = MultiTaskTrainEngine(save_path,
                                        batch_size=kwargs.get('batch_size', 128),
                                        infer_batch_size=kwargs.get('infer_batch_size', 128),
                                        epochs=kwargs.get('epochs', 100),
                                        steps_per_epoch=kwargs.get('steps_per_epoch'),
                                        learning_rate=kwargs.get('learning_rate', 1e-3),
                                        dropout=kwargs.get('dropout', 0.0),
                                        bs_masks=bs_masks,
                                        svd_weight=svd_weight,
                                        loss_epsilon=kwargs.get('loss_epsilon', 0.0),
                                        verbose=kwargs.pop('verbose', 1))

    train_process = Process(target=train_engine,
                            args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid),
                                pretrained_path
                            ))

    train_process.start()
    train_process.join()


def semi_train(data_file,
               label_file,
               save_path,
               pretrained_path=None,
               mask_mode=1,
               learn_svd=False,
               **kwargs):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.threading.set_inter_op_parallelism_threads(4)
    x, y = load_data(data_file, label_file)
    if learn_svd:
        svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    else:
        svd_weight = None

    test_size = kwargs.pop('test_size', 0.1)
    x_unlabel = x[len(y):]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x[:len(y)], y / 120, test_size=test_size)

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    train_engine = SemiTrainEngine(save_path, x_unlabel,
                                   batch_size=kwargs.get('batch_size', 128),
                                   infer_batch_size=kwargs.get('infer_batch_size', 128),
                                   epochs=kwargs.get('epochs', 100),
                                   steps_per_epoch=kwargs.get('steps_per_epoch'),
                                   learning_rate=kwargs.get('learning_rate', 1e-3),
                                   dropout=kwargs.get('dropout', 0.0),
                                   bs_masks=bs_masks,
                                   svd_weight=svd_weight,
                                   loss_epsilon=kwargs.get('loss_epsilon', 0.0),
                                   verbose=kwargs.pop('verbose', 1))

    train_process = Process(target=train_engine,
                            args=(
                                 (x_train, y_train),
                                 (x_valid, y_valid),
                                pretrained_path
                            ))

    train_process.start()
    train_process.join()


def test(data_file,
         label_file,
         model_path,
         result_file=None):

    x, y = load_data(data_file, label_file)
    model = tf.keras.models.load_model(model_path)
    pred = model.predict(x)
    rmse = np.mean(np.sqrt(np.sum((y - pred[:len(y)]) ** 2, axis=-1)))
    print('RMSE: ', round(rmse, 4))

    if result_file is not None:
        np.save(result_file, pred.transpose((1, 0)))


if __name__ == '__main__':
    fire.Fire()
