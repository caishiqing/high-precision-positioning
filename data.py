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
    def __init__(self, total_bs, min_bs=4, max_bs=18):
        self.total_bs = total_bs
        self.min_bs = min_bs
        self.max_bs = max_bs
        self.bs_ids = list(range(total_bs))

    def _mask_bs(self):
        mask = np.zeros(self.total_bs, dtype=np.float32)
        non_zeros = random.sample(self.bs_ids, random.randint(self.min_bs, self.max_bs))
        mask[non_zeros] = 1
        return mask

    def __call__(self, x, y):
        mask = tf.py_function(self._mask_bs, [], tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, tf.newaxis]
        mask = tf.tile(mask, [1, 4, 1, 1])
        mask = tf.reshape(mask, [-1, 1, 1])
        x *= mask
        return x, y


if __name__ == '__main__':
    x = np.random.random((1000, 72, 2, 256)).astype(np.float32)
    y = np.random.random((1000, 2)).astype(np.float32)
    augment = MaskBS(18, 4, 18)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(augment).batch(8)
    for x, y in dataset:
        xx = tf.reshape(x, [8, 72, -1])
        mask = tf.reduce_any(tf.not_equal(xx, 0), axis=-1)
        print(mask)
        pass
