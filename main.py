from sklearn.model_selection import train_test_split
from train import TrainEngine, load_data
from multiprocessing import Process
import numpy as np
import fire


def train(data_file, label_file, save_path,
          pretrained_path=None, **kwargs):

    x, y = load_data(data_file, label_file)
    x = x[:len(y)]
    test_size = kwargs.pop('test_size', 0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size)

    train_engine = TrainEngine(batch_size=kwargs.pop('batch_size', 128),
                               infer_batch_size=kwargs.pop('infer_batch_size', 128),
                               epochs=kwargs.pop('epochs', 100),
                               learning_rate=kwargs.pop('learning_rate', 1e-3),
                               valid_augment_times=kwargs.pop('test_augment_times', 5),
                               dropout=kwargs.pop('dropout', 0.0),
                               min_bs=kwargs.pop('min_bs', 4),
                               max_bs=kwargs.pop('max_bs', 18),
                               mask_rate=kwargs.pop('mask_rate', 0.0))

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
