from random import triangular
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework.smart_cond import smart_cond


bs_masks = [
    [0, 5, 12, 17],
    [0, 5, 12],
    [0, 5, 17],
    [5, 12, 17],
    [0, 12, 17],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [1, 4, 13, 16],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
]


class MultiHeadAttention(layers.Layer):
    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 output_dim=None,
                 dropout=0.0,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert key_dim % num_heads == 0
        self.support_masking = True
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.sqrt_dim = tf.math.sqrt(float(self.key_dim) / self.num_heads)

    def build(self, input_shape):
        if len(input_shape) == 3 and isinstance(input_shape[0], tuple):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape

        self.query = layers.Dense(self.key_dim,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name='query')

        self.key = layers.Dense(self.key_dim,
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                name='key')

        self.value = layers.Dense(self.value_dim if self.value_dim else self.key_dim,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name='value')

        self.out = layers.Dense(self.output_dim if self.output_dim else query_shape[-1],
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                name='out')

        self.drop = layers.Dropout(self.dropout)
        super(MultiHeadAttention, self).build(input_shape)

    def _split_heads(self, x):
        x = tf.split(x, self.num_heads, axis=-1)
        x = tf.stack(x, axis=1)
        return x

    def _merge_heads(self, x):
        batch_size, _, input_length, size_per_head = tf.keras.backend.int_shape(x)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, input_length, self.num_heads * size_per_head])
        x.set_shape((batch_size, input_length, self.num_heads * size_per_head))
        return x

    def _mask_softmax(self, x, mask=None, axis=-1):
        if mask is not None:
            mask = tf.cast(mask, x.dtype)
            x -= (1 - mask) * 1e6
        return tf.nn.softmax(x, axis=axis)

    def _compute_attention(self, q, k, v, attention_mask=None, training=None):
        attention_score = tf.matmul(q, k, transpose_b=True) / self.sqrt_dim
        attention_proba = self._mask_softmax(attention_score, attention_mask)
        attention_proba = self.drop(attention_proba, training=training)
        attention_output = tf.matmul(attention_proba, v)
        return attention_output

    def call(self, inputs, attention_mask=None, training=None):
        query, key, value = inputs
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if attention_mask is not None:
            attention_mask = attention_mask[:, tf.newaxis, :, :]

        attention_output = self._compute_attention(query, key, value,
                                                   attention_mask=attention_mask,
                                                   training=training)

        attention_output = self._merge_heads(attention_output)
        output = self.out(attention_output)
        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3 and isinstance(input_shape[0], tuple):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape

        return query_shape[0], query_shape[1], self.out.units

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update(
            {
                'num_heads': self.num_heads,
                'key_dim': self.key_dim,
                'value_dim': self.value_dim,
                'output_dim': self.output_dim,
                'dropout': self.dropout,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer
            }
        )
        return config


class SelfAttention(MultiHeadAttention):
    def call(self, x, mask=None, training=None):
        if mask is not None:
            attention_mask = tf.logical_and(mask[:, :, tf.newaxis],
                                            mask[:, tf.newaxis, :])
        else:
            attention_mask = None

        return super(SelfAttention, self).call([x, x, x],
                                               attention_mask=attention_mask,
                                               training=training)

    def compute_mask(self, x, mask=None):
        return mask


class AntennaEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        super(AntennaEmbedding, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        _, num_antennas, dim = input_shape
        self.embeddings = self.add_weight(name='{}_embeddings'.format(self.name),
                                          shape=(num_antennas + 1, dim),
                                          dtype=tf.float32,
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.built = True

    def call(self, x, mask=None):
        cls_embed = tf.tile(self.embeddings[tf.newaxis, :1, :], [tf.shape(x)[0], 1, 1])
        ant_embed = self.embeddings[tf.newaxis, 1:, :]
        if mask is not None:
            ant_embed *= tf.cast(mask, ant_embed.dtype)[:, :, tf.newaxis]

        x += ant_embed
        x = tf.concat([cls_embed, x], axis=1)
        return x

    def compute_output_shape(self, input_shape):
        batch, num_antennas, dim = input_shape
        return batch, num_antennas + 1, dim

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        #cls_mask = tf.cast(tf.ones(shape=(tf.shape(mask)[0], 1)), mask.dtype)
        cls_mask = tf.ones_like(mask, dtype=mask.dtype)[:, :1]
        mask = tf.concat([cls_mask, mask], axis=1)
        return mask


# def Conv(x):
#     x = layers.Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]))(x)
#     x1, x2, x3, x4 = layers.Lambda(lambda x: tf.split(x, 4, axis=2))(x)

#     x1 = layers.TimeDistributed(layers.Conv1D(128, 61))(x1)
#     x1 = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x1)

#     x2 = layers.TimeDistributed(layers.Conv1D(64, 61))(x2)
#     x2 = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x2)

#     x3 = layers.TimeDistributed(layers.Conv1D(32, 61))(x3)
#     x3 = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x3)

#     x4 = layers.TimeDistributed(layers.Conv1D(16, 61))(x4)
#     x4 = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x4)

#     x = layers.Concatenate()([x1, x2, x3, x4])
#     x = layers.LayerNormalization()(x)
#     x = layers.Activation('relu')(x)
#     return x


def Residual(fn, res, dropout=0.0):
    x = fn(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([res, x])
    x = layers.LayerNormalization()(x)
    return x


def SVD(x, units=256):
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Masking()(x)
    x = layers.Dense(units, use_bias=False, trainable=False, name='svd')(x)
    return x


class MultiHeadBS(layers.Layer):
    def __init__(self,
                 bs_masks=None,
                 num_bs=18,
                 num_antennas_per_bs=4,
                 **kwargs):
        super(MultiHeadBS, self).__init__(**kwargs)
        if bs_masks is None:
            bs_masks = [list(range(num_bs))]

        self.bs_masks = bs_masks
        self.num_masks = len(bs_masks)
        self.num_bs = num_bs
        self.num_antennas_per_bs = num_antennas_per_bs

    def build(self, input_shape):
        B, S, _, T = input_shape
        assert self.num_bs * self.num_antennas_per_bs == S
        self.T = T

        masks = np.zeros((self.num_masks, self.num_bs), dtype=np.float32)
        for i, bs_mask in enumerate(self.bs_masks):
            masks[i][bs_mask] = 1

        masks = tf.tile(tf.expand_dims(tf.identity(masks), -1), [1, 1, self.num_antennas_per_bs])
        self.masks = tf.reshape(masks, [self.num_masks, -1])
        self.build = True

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def _train():
            rand = tf.cast(tf.random.uniform(tf.shape(inputs)[:1]) * self.num_masks, tf.int32)
            mask = tf.gather(self.masks, rand)[:, :, tf.newaxis, tf.newaxis]
            x = inputs * mask
            return tf.expand_dims(x, 1)

        def _infer():
            # (batch, N, 72, 2, 256)
            masks = self.masks[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
            return tf.expand_dims(inputs, 1) * masks

        output = smart_cond(training, _train, _infer)
        return output

    def get_config(self):
        config = super().get_config()
        config['bs_masks'] = self.bs_masks
        config['num_bs'] = self.num_bs
        config['num_antennas_per_bs'] = self.num_antennas_per_bs
        return config


class MyTimeDistributed(layers.TimeDistributed):
    def __init__(self, layer, num_bs=18, min_bs=3, **kwargs):
        super(MyTimeDistributed, self).__init__(layer, **kwargs)
        self.num_bs = num_bs
        self.min_bs = min_bs

    def compute_mask(self, x, mask=None):
        an_mask = tf.reduce_any(tf.not_equal(x, 0), axis=[3, 4])
        an_mask = tf.stack(tf.split(an_mask, self.num_bs, -1), 2)
        bs_mask = tf.reduce_any(an_mask, -1)
        head_mask = tf.greater_equal(tf.reduce_sum(tf.cast(bs_mask, tf.int32), -1), self.min_bs)
        return head_mask

    def get_config(self):
        config = super().get_config()
        config['num_bs'] = self.num_bs
        config['min_bs'] = self.min_bs
        return config


def build_model(input_shape,
                output_shape=2,
                embed_dim=256,
                hidden_dim=1024,
                num_heads=8,
                num_attention_layers=6,
                dropout=0.0,
                bs_masks=None,
                norm_size=120):

    assert embed_dim % num_heads == 0

    x = layers.Input(shape=input_shape)
    h = SVD(x, embed_dim)
    h = AntennaEmbedding()(h)
    h = layers.Dense(embed_dim)(h)
    h = layers.LayerNormalization()(h)
    h = layers.Activation('relu')(h)

    for _ in range(num_attention_layers):
        h = Residual(SelfAttention(num_heads, embed_dim, dropout=dropout), h, dropout=dropout)
        h = Residual(
            tf.keras.Sequential(
                layers=[
                    layers.Dense(hidden_dim, activation='relu'),
                    layers.Dense(embed_dim)
                ]
            ),
            h,
            dropout=dropout
        )

    cls_h = layers.Lambda(lambda x: x[:, 0, :])(h)
    y = layers.Dense(output_shape, activation='sigmoid', name='pos')(cls_h)

    model = tf.keras.Model(x, y, name='base')
    model_wrapper = tf.keras.Sequential()
    model_wrapper.add(layers.Input(input_shape))
    model_wrapper.add(MultiHeadBS(bs_masks, 18, 4, name='mask')),
    model_wrapper.add(MyTimeDistributed(model, 18, 3, name='wrapper'))
    model_wrapper.add(layers.GlobalAveragePooling1D())
    if norm_size is not None:
        model_wrapper.add(layers.Lambda(lambda x: x * norm_size))

    return model_wrapper


tf.keras.utils.get_custom_objects().update(
    {
        'MultiHeadAttention': MultiHeadAttention,
        'SelfAttention': SelfAttention,
        'AntennaEmbedding': AntennaEmbedding,
        'MultiHeadBS': MultiHeadBS,
        'MyTimeDistributed': MyTimeDistributed
    }
)


def ensemble(models):
    x = layers.Input(shape=models[0].input_shape[1:])
    ys = []
    for i, model in enumerate(models):
        model._name = '{}_{}'.format(model._name, i)
        ys.append(model(x))

    y = layers.Average()(ys)
    model = tf.keras.Model(x, y)
    return model


def Model_1(input_shape, output_shape, kfold=1):
    models = [build_model(input_shape, output_shape) for _ in range(kfold)]
    if kfold == 1:
        return models[0]

    return ensemble(models)


if __name__ == '__main__':
    model = Model_1((72, 2, 256), 2)
    model.load_weights('modelSubmit_1.h5')
    model.save('modelSubmit_1.h5', include_optimizer=False)
