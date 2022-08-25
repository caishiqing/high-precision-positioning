from sklearn.model_selection import train_test_split
from train import TrainEngine,  PretrainEngine, load_data
from sklearn.decomposition import PCA, TruncatedSVD
from modelDesign_1 import bs_masks as masks1
from modelDesign_2 import bs_masks as masks2
from multiprocessing import Process
import numpy as np
import fire


def train(data_file, label_file, save_path,
          pretrained_path=None, mask_mode=1, **kwargs):

    x, y = load_data(data_file, label_file)
    svd_weight = TruncatedSVD(256).fit(x.reshape([len(x) * 72, -1])).components_.T
    x = x[:len(y)]
    test_size = kwargs.pop('test_size', 0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size)
    del x, y

    if mask_mode == 1:
        bs_masks = masks1
    elif mask_mode == 2:
        bs_masks = masks2
    print(bs_masks)

    train_engine = TrainEngine(batch_size=kwargs.pop('batch_size', 128),
                               infer_batch_size=kwargs.pop('infer_batch_size', 128),
                               epochs=kwargs.pop('epochs', 100),
                               learning_rate=kwargs.pop('learning_rate', 1e-3),
                               dropout=kwargs.pop('dropout', 0.0),
                               masks=bs_masks,
                               svd_weight=svd_weight)

    train_process = Process(target=train_engine,
                            args=(
                                (x_train, y_train),
                                (x_valid, y_valid),
                                save_path,
                                pretrained_path,
                                kwargs.pop('verbose', 1)
                            ))
    train_process.start()
    train_process.join()


def pretrain(data_file, label_file, save_path,
             pretrained_path=None, **kwargs):

    x, _ = load_data(data_file, label_file)
    u, s, v = np.linalg.svd(np.reshape(x[:, :, 0, :], [len(x) * 72, 256]), False)
    yi = u[:, :32]
    u, s, v = np.linalg.svd(np.reshape(x[:, :, 1, :], [len(x) * 72, 256]), False)
    yj = u[:, :32]
    y = np.concatenate([yi, yj], axis=-1).reshape([len(x), 72, 64])
    test_size = kwargs.pop('test_size', 0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size)
    del x

    train_engine = PretrainEngine(batch_size=kwargs.pop('batch_size', 128),
                                  infer_batch_size=kwargs.pop('infer_batch_size', 128),
                                  epochs=kwargs.pop('epochs', 100),
                                  learning_rate=kwargs.pop('learning_rate', 1e-3),
                                  dropout=kwargs.pop('dropout', 0.0))

    train_process = Process(target=train_engine,
                            args=(
                                (x_train, y_train),
                                (x_valid, y_valid),
                                save_path,
                                pretrained_path,
                                kwargs.pop('verbose', 1)
                            ))
    train_process.start()
    train_process.join()


def test(data_file, label_file, model_path):
    from modelDesign_1 import build_model

    x, y = load_data(data_file, label_file)
    x = x[:len(y)]
    model = build_model(input_shape=x.shape[1:])
    model.load_weights(model_path)
    pred = model.predict(x)
    rmse = np.mean(np.math.sqrt(np.sum((y - pred) ** 2, axis=-1)))
    return rmse


if __name__ == '__main__':
    fire.Fire()
