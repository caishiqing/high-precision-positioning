from modelDesign_1 import build_model
from optimizer import AdamWarmup
import tensorflow as tf
import numpy as np


def load_data(x_file, y_file):
    CIR = np.load(x_file).astype(np.float32)
    x = CIR.transpose((2, 3, 0, 1))
    POS = np.load(y_file).astype(np.float32)
    y = POS.transpose((1, 0))
    return x, y / 120


# class MaskBS(object):
#     def __init__(self, total_bs, mask_rate=0.0):
#         self.total_bs = total_bs
#         self.mask_rate = mask_rate
#         mask = np.zeros(total_bs, dtype=np.float32)
#         mask[[0, 5, 12, 17]] = 1
#         mask = tf.constant(mask)
#         # mask = mask[:, tf.newaxis, tf.newaxis, tf.newaxis]
#         # mask = tf.tile(mask, [1, 4, 1, 1])
#         # self._mask = tf.reshape(mask, [-1, 1, 1])
#         mask = mask[:, tf.newaxis, tf.newaxis]
#         mask = tf.tile(mask, [1, 4, 1])
#         self._mask = tf.reshape(mask, [-1, 1])

#     def __call__(self, x, y):
#         x = tf.cond(tf.random.uniform([]) < self.mask_rate,
#                     lambda: x * self._mask, lambda: x)
#         return x, y


class MaskBS(object):
    def __init__(self, total_bs=18, num_antennas_per_bs=4, masks=None):
        if masks is None:
            masks = [list(range(total_bs))]

        self.num_masks = len(masks)
        self.masks = np.zeros((self.num_masks, total_bs), dtype=np.float32)
        for i, mask in enumerate(masks):
            self.masks[i][mask] = 1

        self.masks = tf.constant(self.masks)[:, :, tf.newaxis, tf.newaxis, tf.newaxis]
        self.masks = tf.tile(self.masks, [1, 1, num_antennas_per_bs, 1, 1])

    def __call__(self, x, y):
        rand = tf.cast(tf.random.uniform([]) * self.num_masks, tf.int32)
        mask = self.masks[rand]
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
                 dropout=0.0,
                 masks=None,
                 svd_weight=None):

        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.drop_remainder = False
        self.augment = MaskBS(18, 4, masks)
        self.svd_weight = svd_weight

    def _init_environ(self):
        # Build distribute strategy on gpu or tpu
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            self.drop_remainder = True
        except:
            print("Runing on gpu or cpu")
            strategy = tf.distribute.get_strategy()

        return strategy

    def __call__(self, train_data, valid_data,
                 save_path, pretrained_path=None, verbose=1):

        stratagy = self._init_environ()

        x_train_shape = train_data[0].shape
        x_valid_shape = valid_data[0].shape

        autotune = tf.data.experimental.AUTOTUNE
        train_data = tf.data.Dataset.from_tensor_slices(
            train_data).map(self.augment, autotune).batch(self.batch_size, self.drop_remainder)
        valid_data = tf.data.Dataset.from_tensor_slices(
            valid_data).map(self.augment, autotune).batch(x_valid_shape[0])
        valid_data = list(valid_data)[0]

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

            model = build_model(x_train_shape[1:], 2,
                                dropout=self.dropout,
                                svd_weight=self.svd_weight)
            if pretrained_path is not None:
                print("Load pretrained weights from {}".format(pretrained_path))
                model.load_weights(pretrained_path)

            model.compile(optimizer=optimizer, loss=tf.keras.losses.mae)
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


class PretrainEngine(TrainEngine):
    def __call__(self, train_data, valid_data,
                 save_path, pretrained_path=None, verbose=1):

        strategy = self._init_environ()
        x_train_shape = train_data[0].shape
        x_valid_shape = valid_data[0].shape

        autoturn = tf.data.AUTOTUNE
        augment = RandomMaskBS(18)
        train_data = tf.data.Dataset.from_tensor_slices(
            train_data).map(augment, autoturn).batch(self.batch_size,
                                                     drop_remainder=self.drop_remainder)
        valid_data = tf.data.Dataset.from_tensor_slices(
            valid_data).map(augment, autoturn).batch(x_valid_shape[0])
        valid_data = list(valid_data)[0]

        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode='min',
                                                        monitor='val_loss')

        with strategy.scope():
            total_steps = x_train_shape[0] // self.batch_size * self.epochs
            optimizer = AdamWarmup(warmup_steps=int(total_steps * 0.1),
                                   decay_steps=total_steps-int(total_steps * 0.1),
                                   initial_learning_rate=self.learning_rate)

            # Train mask-bs model weights
            model, mbs_model = build_model(x_train_shape[1:], 2, dropout=self.dropout)
            if pretrained_path is not None:
                print("Load pretrained weights from {}".format(pretrained_path))
                model.load_weights(pretrained_path)

            mbs_model.compile(optimizer=optimizer,
                              loss=tf.keras.losses.mae)
            mbs_model.summary()
            mbs_model.fit(x=train_data,
                          epochs=self.epochs,
                          validation_data=valid_data,
                          validation_batch_size=self.infer_batch_size,
                          callbacks=[checkpoint],
                          verbose=verbose,
                          shuffle=True)

        print(checkpoint.best)
        mbs_model.load_weights(save_path)
        model.save(save_path)
        return model


if __name__ == '__main__':
    x = np.random.random((2, 16, 2, 4)).astype(np.float32)
    y = np.random.random((2, 2)).astype(np.float32)
    augment = MaskBS(8, 2, [[1, 2, 4], [0, 6, 7], [3, 4, 5]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(augment).repeat()
    for xx, yy in dataset:
        print(xx, '\n')
