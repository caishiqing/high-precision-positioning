from sklearn.model_selection import train_test_split
from train import TrainEngine, load_data
from multiprocessing import Process
import tensorflow as tf
import fire


def train(data_file, label_file, save_path,
          pretrained_path=None, **kwargs):

    x, y = load_data(data_file, label_file)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size)

    train_engine = TrainEngine(batch_size=kwargs.pop('batch_size', 128),
                               infer_batch_size=kwargs.pop('infer_batch_size', 128),
                               epochs=kwargs.pop('epochs', 100),
                               learning_rate=kwargs.pop('learning_rate', 1e-3),
                               valid_augment_times=kwargs.pop('test_augment_times', 5))

    train_process = Process(target=train_engine,
                            args=(
                                (x_train, y_train),
                                (x_valid, y_valid),
                                save_path,
                                pretrained_path
                            ))
    train_process.start()
    train_process.join()


if __name__ == '__main__':
    fire.Fire()
