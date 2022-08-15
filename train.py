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
    def __init__(self, total_bs, min_bs=4, max_bs=18, mask_rate=0.0):
        self.total_bs = total_bs
        self.min_bs = min_bs
        self.max_bs = max_bs
        self.mask_rate = mask_rate
        self.bs_ids = list(range(total_bs))
        self.zeros = list(set(range(18)) - set([0, 5, 12, 17]))

    def _mask_bs(self):
        mask = np.ones(self.total_bs, dtype=np.float32)
        if self.mask_rate > 0 and random.random() < self.mask_rate:
            # num_zeros = self.total_bs - random.randint(self.min_bs, self.max_bs)
            # zeros = random.sample(self.bs_ids, num_zeros)
            zeros = self.zeros
            mask[zeros] = 0

        return mask

    def __call__(self, x, y):
        mask = tf.py_function(self._mask_bs, [], tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, tf.newaxis]
        mask = tf.tile(mask, [1, 4, 1, 1])
        mask = tf.reshape(mask, [-1, 1, 1])
        x *= mask
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
                 valid_augment_times=1,
                 dropout=0.0,
                 min_bs=4,
                 max_bs=18,
                 mask_rate=0.0):

        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.valid_augment_times = int(valid_augment_times)
        self.dropout = float(dropout)
        self.min_bs = int(min_bs)
        self.max_bs = int(max_bs)
        self.mask_rate = float(mask_rate)

    def __call__(self, train_data, valid_data,
                 save_path, pretrained_path=None, verbose=1):

        x_train, y_train = train_data
        x_valid, y_valid = valid_data
        x_valid = np.vstack([x_valid] * self.valid_augment_times)
        y_valid = np.vstack([y_valid] * self.valid_augment_times)

        autoturn = tf.data.AUTOTUNE
        augment = MaskBS(18, self.min_bs, self.max_bs, self.mask_rate)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            train_data).map(augment, num_parallel_calls=autoturn).batch(self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            valid_data).map(augment, num_parallel_calls=autoturn).batch(len(x_valid))
        valid_dataset = list(valid_dataset)[0]

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='min',
                                                        monitor='val_loss')

        total_steps = len(x_train) // self.batch_size * self.epochs
        optimizer = AdamWarmup(warmup_steps=int(total_steps * 0.1),
                               decay_steps=total_steps-int(total_steps * 0.1),
                               initial_learning_rate=self.learning_rate)

        model = build_model(x_train.shape[1:], 2, dropout=self.dropout)
        if pretrained_path is not None:
            print("Load pretrained weights from {}".format(pretrained_path))
            model.load_weights(pretrained_path)

        model.compile(optimizer=optimizer, loss=euclidean_loss)
        model.summary()
        model.fit(x=train_dataset,
                  epochs=self.epochs,
                  validation_data=valid_dataset,
                  validation_batch_size=self.infer_batch_size,
                  callbacks=[checkpoint],
                  verbose=verbose)

        print(checkpoint.best)
        model.load_weights(save_path)
        return model


if __name__ == '__main__':
    x = np.random.random((1000, 72, 2, 4)).astype(np.float32)
    y = np.random.random((1000, 2)).astype(np.float32)
    augment = MaskBS(18, 1, 2, 0.5)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(augment).batch(len(x))
    xx, yy = list(dataset)[0]
    pass
