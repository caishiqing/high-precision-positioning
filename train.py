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


def clip_loss(loss_fn, epsilon=0):
    if epsilon == 0:
        return loss_fn

    def _loss_fn(y_true, y_pred):
        loss = loss_fn(y_true, y_pred)
        _clip_loss = tf.clip_by_value(loss, epsilon, 1e3)
        return _clip_loss

    return _loss_fn


class TrainEngine:
    def __init__(self,
                 batch_size,
                 infer_batch_size,
                 epochs=100,
                 steps_per_epoch=None,
                 learning_rate=1e-3,
                 dropout=0.0,
                 masks=None,
                 svd_weight=None,
                 loss_epsilon=0):

        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.epochs = int(epochs)
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.drop_remainder = False
        self.augment = MaskBS(18, 4, masks)
        self.svd_weight = svd_weight
        self.loss_epsilon = float(loss_epsilon)

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

    def _compute_warmup_steps(self, num_samples):
        if self.steps_per_epoch is not None:
            total_steps = self.steps_per_epoch * self.epochs
            self.drop_remainder = True
        else:
            total_steps = num_samples // self.batch_size

        warmup_steps = int(total_steps * 0.1)
        decay_steps = total_steps - warmup_steps
        return warmup_steps, decay_steps

    def __call__(self, train_data, valid_data, save_path,
                 pretrained_path=None, verbose=1):

        strategy = self._init_environ()
        x_train_shape = train_data[0].shape
        x_valid_shape = valid_data[0].shape

        autotune = tf.data.experimental.AUTOTUNE
        train_data = tf.data.Dataset.from_tensor_slices(
            train_data).map(self.augment, autotune)
        if self.steps_per_epoch is not None:
            train_data = train_data.shuffle(1000)
        else:
            train_data = train_data.batch(self.batch_size, self.drop_remainder)

        valid_data = tf.data.Dataset.from_tensor_slices(
            valid_data).map(self.augment, autotune).batch(x_valid_shape[0])
        valid_data = list(valid_data)[0]

        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='min',
                                                        monitor='val_loss')

        with strategy.scope():
            warmup_steps, decay_steps = self._compute_warmup_steps(x_train_shape[0])
            optimizer = AdamWarmup(warmup_steps=warmup_steps,
                                   decay_steps=decay_steps,
                                   initial_learning_rate=self.learning_rate)

            model = build_model(x_train_shape[1:], 2, dropout=self.dropout)
            svd_layer = model.layers[3]
            if pretrained_path is not None:
                print("Load pretrained weights from {}".format(pretrained_path))
                model.load_weights(pretrained_path)
            if self.svd_weight is not None:
                print('Load svd weight!')
                svd_layer.set_weights([self.svd_weight])

            svd_layer.trainable = False
            model.compile(optimizer=optimizer,
                          loss=clip_loss(tf.keras.losses.mae,
                                         self.loss_epsilon))
            model.summary()
            model.fit(x=train_data,
                      epochs=self.epochs,
                      steps_per_epoch=self.steps_per_epoch,
                      validation_data=valid_data,
                      validation_batch_size=self.infer_batch_size,
                      callbacks=[checkpoint],
                      verbose=verbose)

        print(checkpoint.best)
        model.load_weights(save_path)
        return model


if __name__ == '__main__':
    x = np.random.random((2, 16, 2, 4)).astype(np.float32)
    y = np.random.random((2, 2)).astype(np.float32)
    augment = MaskBS(8, 2, [[1, 2, 4], [0, 6, 7], [3, 4, 5]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(augment).repeat()
    for xx, yy in dataset:
        print(xx, '\n')
