from enum import auto
from tabnanny import check
from optimizer import AdamWarmup
from modelDesign_1 import build_model
import tensorflow as tf
import numpy as np
import random


def load_data(x_file, y_file):
    CIR = np.load(x_file).astype(np.float32)
    x = CIR.transpose((2, 3, 0, 1))
    POS = np.load(y_file).astype(np.float32)
    y = POS.transpose((1, 0))
    return x, y


class MaskBS(object):
    def __init__(self, total_bs, mask_rate=0.0):
        self.total_bs = total_bs
        self.mask_rate = mask_rate
        mask = np.zeros(total_bs, dtype=np.float32)
        mask[[0, 5, 12, 17]] = 1
        mask = tf.identity(mask)
        mask = mask[:, tf.newaxis, tf.newaxis, tf.newaxis]
        mask = tf.tile(mask, [1, 4, 1, 1])
        self._mask = tf.reshape(mask, [-1, 1, 1])

    def __call__(self, x, y):
        x = tf.where(random.random() < self.mask_rate, x * self._mask, x)
        return x, y


def euclidean_loss(y_true, y_pred):
    distance = tf.math.sqrt(tf.reduce_sum(tf.pow(y_true-y_pred, 2), axis=-1))
    return tf.reduce_mean(distance)


class TrainEngine:
    def __init__(self,
                 batch_size,
                 infer_batch_size,
                 epochs=100,
                 learning_rate=1e-3,
                 dropout=0.0,
                 mask_rate=0.0):

        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.mask_rate = float(mask_rate)

    def __call__(self, train_data, valid_data,
                 save_path, pretrained_path=None, verbose=1):

        # Build distribute strategy on gpu or tpu
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            drop_remainder = True
        except:
            print("Runing on gpu or cpu")
            strategy = tf.distribute.get_strategy()
            drop_remainder = False

        x_train_shape = train_data[0].shape
        x_valid_shape = valid_data[0].shape
        train_data = tf.data.Dataset.from_tensor_slices(train_data)
        if self.mask_rate > 0:
            print("Masking BS ...")
            autoturn = tf.data.AUTOTUNE
            augment = MaskBS(18, self.mask_rate)
            train_data = train_data.map(augment, autoturn)
            valid_data = tf.data.Dataset.from_tensor_slices(
                valid_data).map(augment, autoturn)
            valid_data = list(valid_data.batch(x_valid_shape[0]))[0]

        train_data = train_data.batch(self.batch_size, drop_remainder=drop_remainder)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='min',
                                                        monitor='val_loss')

        with strategy.scope():
            total_steps = x_train_shape[0] // self.batch_size * self.epochs
            optimizer = AdamWarmup(warmup_steps=int(total_steps * 0.1),
                                   decay_steps=total_steps-int(total_steps * 0.1),
                                   initial_learning_rate=self.learning_rate)

            model = build_model(x_train_shape[1:], 2, dropout=self.dropout)
            if pretrained_path is not None:
                print("Load pretrained weights from {}".format(pretrained_path))
                model.load_weights(pretrained_path)

            model.compile(optimizer=optimizer, loss=euclidean_loss)
            model.summary()
            model.fit(x=train_data,
                      epochs=self.epochs,
                      validation_data=valid_data,
                      validation_batch_size=self.infer_batch_size,
                      callbacks=[checkpoint],
                      verbose=verbose,
                      shuffle=True)

        print(checkpoint.best)
        model.load_weights(save_path)
        return model


class Checkpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, start_checkpoint_epoch=0, **kwargs):
        super(Checkpoint, self).__init__(filepath, **kwargs)
        self.start_checkpoint_epoch = start_checkpoint_epoch

    def _save_model(self, epoch, logs):
        if epoch >= self.start_checkpoint_epoch:
            super(Checkpoint, self)._save_model(epoch, logs)


if __name__ == '__main__':
    x = np.random.random((1000, 72, 2, 4)).astype(np.float32)
    y = np.random.random((1000, 2)).astype(np.float32)
    augment = MaskBS(18, 1, 2, 0.5)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(augment).batch(len(x))
    xx, yy = list(dataset)[0]
    pass
