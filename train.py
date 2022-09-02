from modelDesign_1 import build_model
from optimizer import AdamWarmup
import tensorflow as tf
import numpy as np


def load_data(x_file, y_file):
    CIR = np.load(x_file).astype(np.float32)
    x = CIR.transpose((2, 3, 0, 1))
    POS = np.load(y_file).astype(np.float32)
    y = POS.transpose((1, 0))
    return x, y


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
                 save_path,
                 batch_size,
                 infer_batch_size,
                 epochs=100,
                 learning_rate=1e-3,
                 dropout=0.0,
                 bs_masks=None,
                 svd_weight=None,
                 loss_epsilon=0):

        self.save_path = save_path
        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.drop_remainder = False
        self.augment = MaskBS(18, 4, bs_masks)
        self.svd_weight = svd_weight
        self.loss_epsilon = float(loss_epsilon)

        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                             save_best_only=True,
                                                             save_weights_only=False,
                                                             mode='min',
                                                             monitor='val_loss')

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

    @staticmethod
    def _shuffle_data(x, y):
        ids = list(range(len(x)))
        np.random.shuffle(ids)
        x = x[ids]
        y = y[ids]
        return x, y

    def _compute_warmup_steps(self, num_samples):
        total_steps = num_samples // self.batch_size * self.epochs
        warmup_steps = int(total_steps * 0.1)
        decay_steps = total_steps - warmup_steps
        return warmup_steps, decay_steps

    def _prepare_dataset(self, train_data, valid_data, repeat_data_times=1):
        if repeat_data_times > 1:
            x_train, y_train = train_data
            x_train = np.vstack([x_train] * repeat_data_times)
            y_train = np.vstack([y_train] * repeat_data_times)
            train_data = self._shuffle_data(x_train, y_train)

        autotune = tf.data.experimental.AUTOTUNE
        train_dataset = tf.data.Dataset.from_tensor_slices(
            train_data).map(self.augment, autotune).batch(self.batch_size,
                                                          self.drop_remainder)
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            valid_data).map(self.augment, autotune).batch(valid_data[0].shape[0])
        valid_dataset = list(valid_dataset)[0]
        return train_dataset, valid_dataset

    def __call__(self,
                 train_data,
                 valid_data,
                 repeat_data_times=1,
                 pretrained_path=None,
                 verbose=1,
                 model=None,):

        strategy = self._init_environ()
        x_train_shape = train_data[0].shape
        train_dataset, valid_dataset = self._prepare_dataset(train_data, valid_data,
                                                             repeat_data_times=repeat_data_times)

        with strategy.scope():
            warmup_steps, decay_steps = self._compute_warmup_steps(x_train_shape[0])
            optimizer = AdamWarmup(warmup_steps=warmup_steps,
                                   decay_steps=decay_steps,
                                   initial_learning_rate=self.learning_rate)

            if model is None:
                model = build_model(x_train_shape[1:], 2, dropout=self.dropout)
                svd_layer = model.get_layer('svd')
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
            model.fit(x=train_dataset,
                      epochs=self.epochs,
                      validation_data=valid_dataset,
                      validation_batch_size=self.infer_batch_size,
                      callbacks=[self.checkpoint],
                      verbose=verbose)

        print(self.checkpoint.best)
        model.load_weights(self.save_path)
        return model


class SemiTrainEngine(TrainEngine):
    def __init__(self,
                 save_path,
                 batch_size,
                 infer_batch_size,
                 unlabel_data,
                 **kwargs):

        super(SemiTrainEngine, self).__init__(save_path,
                                              batch_size,
                                              infer_batch_size,
                                              **kwargs)
        self.unlabel_data = unlabel_data

    def __call__(self, train_data, valid_data,
                 repeat_data_times=1,
                 pretrained_path=None,
                 verbose=1):

        model = super(SemiTrainEngine, self).__call__(train_data, valid_data,
                                                      repeat_data_times=repeat_data_times,
                                                      pretrained_path=pretrained_path,
                                                      verbose=verbose)

        y_pred = model.predict(self.unlabel_data, batch_size=self.infer_batch_size)
        semi_x = np.vstack([train_data[0], self.unlabel_data])
        semi_y = np.vstack([train_data[1], y_pred])
        semi_data = self._shuffle_data(semi_x, semi_y)

        self.learning_rate *= 0.8
        self.loss_epsilon = 0.008
        model = super(SemiTrainEngine, self).__call__(semi_data, valid_data,
                                                      verbose=verbose,
                                                      model=model)

        model = super(SemiTrainEngine, self).__call__(train_data, valid_data,
                                                      repeat_data_times=repeat_data_times,
                                                      verbose=verbose,
                                                      model=model)

        y_pred = model.predict(self.unlabel_data, batch_size=self.infer_batch_size)
        semi_x = np.vstack([train_data[0], self.unlabel_data])
        semi_y = np.vstack([train_data[1], y_pred])
        semi_data = self._shuffle_data(semi_x, semi_y)

        self.learning_rate *= 0.8
        self.loss_epsilon = 0.004
        model = super(SemiTrainEngine, self).__call__(semi_data, valid_data,
                                                      verbose=verbose,
                                                      model=model)

        model = super(SemiTrainEngine, self).__call__(train_data, valid_data,
                                                      repeat_data_times=repeat_data_times,
                                                      verbose=verbose,
                                                      model=model)


if __name__ == '__main__':
    x = np.random.random((100, 16, 2, 4)).astype(np.float32)
    y = np.random.random((100, 2)).astype(np.float32)
    y[:50] = 0
    augment = MaskBS(8, 2, [[1, 2, 4], [0, 6, 7], [3, 4, 5]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(augment).shuffle(100).batch(4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((16, 2, 4)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2)
        ]
    )
