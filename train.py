from modelDesign_1 import build_model
from optimizer import AdamWarmup
import tensorflow as tf
import numpy as np
import types


def load_data(x_file, y_file):
    CIR = np.load(x_file).astype(np.float32)
    x = CIR.transpose((2, 3, 0, 1))
    POS = np.load(y_file).astype(np.float32)
    y = POS.transpose((1, 0))
    return x, y


class MaskBS(object):
    def __init__(self,
                 total_bs=18,
                 num_antennas_per_bs=4,
                 masks=None):

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
        mx = x * mask
        return mx, y


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


def mask_loss(loss_fn):
    def _loss_fn(y_true, y_pred):
        mask = tf.reduce_any(tf.greater(y_true, 0), axis=-1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return loss_fn(y_true, y_pred)

    return _loss_fn


def compare_loss(pos1, pos2):
    label = tf.eye(tf.shape(pos1)[0])
    p1 = tf.expand_dims(pos1, 1)
    p2 = tf.expand_dims(pos2, 0)
    dist = tf.sqrt(tf.reduce_sum(tf.pow(p1 - p2, 2), -1))
    pd = dist[tf.equal(label, 1)]
    nd = dist[tf.equal(label, 0)]

    pos_loss = tf.reduce_logsumexp(pd)
    cmp_loss = tf.nn.softplus(tf.reduce_logsumexp(pd + 1e-3) + tf.reduce_logsumexp(-nd))
    return pos_loss, cmp_loss


def compile(cls, optimizer, loss, **kwargs):
    loss = mask_loss(loss)
    cls.compile(optimizer, loss=loss, **kwargs)


def train_step(cls, data):
    pass


def save_model(cls, filepath, **kwargs):
    kwargs["include_optimizer"] = False
    tf.keras.models.save_model(cls, filepath, **kwargs)


def build_train_model(model):
    x = tf.keras.layers.Input(shape=model.input_shape[1:])
    y = model(x)
    y1 = model(x)
    train_model = tf.keras.Model(x, y)
    train_model.add_loss(compare_loss(y, y1))
    train_model.save = types.MethodType(save_model, train_model)
    return train_model


class TrainEngine:
    def __init__(self,
                 save_path,
                 batch_size=128,
                 infer_batch_size=128,
                 epochs=100,
                 steps_per_epoch=None,
                 learning_rate=1e-3,
                 dropout=0.0,
                 bs_masks=None,
                 svd_weight=None,
                 loss_epsilon=0,
                 monitor='val_loss',
                 verbose=1):

        self.save_path = save_path
        self.batch_size = int(batch_size)
        self.infer_batch_size = int(infer_batch_size)
        self.epochs = int(epochs)
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.drop_remainder = False
        self.bs_masks = bs_masks
        self.svd_weight = svd_weight
        self.loss_epsilon = float(loss_epsilon)
        self.verbose = verbose

        self.autotune = tf.data.experimental.AUTOTUNE
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                             save_best_only=True,
                                                             save_weights_only=False,
                                                             mode='min',
                                                             monitor=monitor)

    def _get_strategy(self):
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
        if self.steps_per_epoch is not None:
            total_steps = self.steps_per_epoch * self.epochs
        else:
            total_steps = num_samples // self.batch_size * self.epochs

        warmup_steps = int(total_steps * 0.1)
        decay_steps = total_steps - warmup_steps
        return warmup_steps, decay_steps

    def _prepare_train_dataset(self, train_data):
        num_samples = len(train_data[0])
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        if self.steps_per_epoch is not None:
            train_dataset = train_dataset.repeat().shuffle(num_samples, reshuffle_each_iteration=True)

        train_dataset = train_dataset.batch(self.batch_size, self.drop_remainder)
        return train_dataset

    def _prepare_valid_dataset(self, valid_data):
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            valid_data).map(self.augment, self.autotune).batch(valid_data[0].shape[0])
        valid_dataset = list(valid_dataset)[0]
        return valid_dataset

    def _train(self,
               train_data,
               valid_data,
               pretrained_path=None):

        strategy = self._get_strategy()
        x_train_shape = train_data[0].shape
        train_dataset = self._prepare_train_dataset(train_data)
        valid_dataset = valid_data

        with strategy.scope():
            warmup_steps, decay_steps = self._compute_warmup_steps(x_train_shape[0])
            optimizer = AdamWarmup(warmup_steps=warmup_steps,
                                   decay_steps=decay_steps,
                                   initial_learning_rate=self.learning_rate)

            model = build_model(x_train_shape[1:], 2,
                                dropout=self.dropout,
                                bs_masks=self.bs_masks)
            model.save = types.MethodType(save_model, model)
            # svd_layer = model.get_layer('svd')
            svd_layer = model.get_layer('wrapper').layer.get_layer('svd')

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
                      steps_per_epoch=self.steps_per_epoch,
                      validation_data=valid_dataset,
                      validation_batch_size=self.infer_batch_size,
                      callbacks=[self.checkpoint],
                      verbose=self.verbose)

        print(self.checkpoint.best)
        model.load_weights(self.save_path)
        return model

    def __call__(self, *args, **kwargs):
        return self._train(*args, **kwargs)


if __name__ == '__main__':
    x = np.random.random((100, 16, 2, 4)).astype(np.float32)
    y = np.random.random((100, 2)).astype(np.float32)
    augment = MaskBS(8, 2, [[1, 2, 4], [0, 6, 7], [3, 4, 5]])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(augment).shuffle(100).batch(4)
    for x, (y, h) in dataset:
        print(x, y, h)
        pass
