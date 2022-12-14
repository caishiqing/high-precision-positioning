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


def Residual(fn, res, dropout=0.0):
    x = fn(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([res, x])
    x = layers.LayerNormalization()(x)
    return x


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


class BSEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        super(BSEmbedding, self).__init__(**kwargs)
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

        cls_mask = tf.ones_like(mask, dtype=mask.dtype)[:, :1]
        mask = tf.concat([cls_mask, mask], axis=1)
        return mask


class BSDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def _train():
            x = super(BSDropout, self).call(inputs, training=True)
            keep_prob = 1 - self.rate
            x *= keep_prob
            return x

        return smart_cond(training, _train, lambda: inputs)


class MultiHeadBS(layers.Layer):
    def __init__(self,
                 bs_masks=None,
                 num_bs=18,
                 **kwargs):
        super(MultiHeadBS, self).__init__(**kwargs)
        if bs_masks is None:
            bs_masks = [list(range(num_bs))]

        self.bs_masks = bs_masks
        self.num_masks = len(bs_masks)
        self.num_bs = num_bs

    def build(self, input_shape):
        masks = np.zeros((self.num_masks, self.num_bs), dtype=np.float32)
        for i, bs_mask in enumerate(self.bs_masks):
            masks[i][bs_mask] = 1

        self.masks = tf.identity(masks)
        self.build = True

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def _train():
            rand = tf.cast(tf.random.uniform(tf.shape(inputs)[:1]) * self.num_masks, tf.int32)
            mask = tf.gather(self.masks, rand)[:, :, tf.newaxis]
            x = inputs * mask
            return tf.expand_dims(x, 1)

        def _infer():
            # (batch, N, 18, 256)
            masks = self.masks[tf.newaxis, :, :, tf.newaxis]
            return tf.expand_dims(inputs, 1) * masks

        output = smart_cond(training, _train, _infer)
        return output

    def get_config(self):
        config = super().get_config()
        config['bs_masks'] = self.bs_masks
        config['num_bs'] = self.num_bs
        return config


class MyTimeDistributed(layers.TimeDistributed):
    def __init__(self, layer, num_bs=18, min_bs=3, **kwargs):
        super(MyTimeDistributed, self).__init__(layer, **kwargs)
        self.num_bs = num_bs
        self.min_bs = min_bs

    def compute_mask(self, x, mask=None):
        bs_mask = tf.reduce_any(tf.not_equal(x, 0), axis=3)
        head_mask = tf.greater_equal(tf.reduce_sum(tf.cast(bs_mask, tf.int32), -1), self.min_bs)
        return head_mask

    def get_config(self):
        config = super().get_config()
        config['num_bs'] = self.num_bs
        config['min_bs'] = self.min_bs
        return config


def build_base_model(embed_dim, hidden_dim,
                     output_shape=2,
                     num_bs=18,
                     dropout=0.0,
                     num_heads=8,
                     num_attention_layers=5):

    x = layers.Input(shape=(num_bs, embed_dim))
    h = BSDropout(dropout)(x)
    h = layers.Masking()(h)
    h = BSEmbedding()(h)
    h = layers.Dense(embed_dim)(h)
    h = layers.LayerNormalization()(h)
    h = layers.Activation('tanh')(h)

    for _ in range(num_attention_layers):
        h = Residual(SelfAttention(num_heads, embed_dim), h)
        h = Residual(
            tf.keras.Sequential(
                layers=[
                    layers.Dense(hidden_dim, activation='relu'),
                    layers.Dense(embed_dim)
                ]
            ),
            h
        )

    h = layers.Lambda(lambda x: x[:, 0, :])(h)
    y = layers.Dense(output_shape, activation='sigmoid')(h)
    model = tf.keras.Model(x, y)
    return model


def build_model(input_shape,
                output_shape=2,
                embed_dim=320,
                hidden_dim=800,
                num_heads=8,
                num_attention_layers=5,
                dropout=0.0,
                bs_masks=None,
                num_bs=18,
                norm_size=[120, 60]):

    assert embed_dim % num_heads == 0
    num_antenna_per_bs = input_shape[0] // num_bs
    norm_size = np.asarray([norm_size], dtype=np.float32)

    preprocess = tf.keras.Sequential(name='svd')
    preprocess.add(layers.TimeDistributed(layers.Flatten()))
    preprocess.add(layers.Dense(embed_dim // num_antenna_per_bs, use_bias=False, trainable=False))
    preprocess.add(layers.Reshape([num_bs, -1]))

    base_model = build_base_model(embed_dim, hidden_dim,
                                  output_shape=output_shape,
                                  num_bs=num_bs,
                                  dropout=dropout,
                                  num_heads=num_heads,
                                  num_attention_layers=num_attention_layers)

    augment_wrapper = tf.keras.Sequential(name='augment_wrapper')
    augment_wrapper.add(layers.Input(shape=(num_bs, embed_dim)))
    augment_wrapper.add(MultiHeadBS(bs_masks, num_bs, name='mask')),
    augment_wrapper.add(MyTimeDistributed(base_model, num_bs, 3, name='wrapper'))
    augment_wrapper.add(layers.GlobalAveragePooling1D())

    x = layers.Input(shape=input_shape)
    h = preprocess(x)
    y_ = augment_wrapper(h)
    y = layers.Lambda(lambda x: x * tf.identity(norm_size), name='pos')(y_)
    model = tf.keras.Model(x, y)

    return model


tf.keras.utils.get_custom_objects().update(
    {
        'MultiHeadAttention': MultiHeadAttention,
        'SelfAttention': SelfAttention,
        'BSEmbedding': BSEmbedding,
        'BSDropout': BSDropout,
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
    models = [build_model(input_shape, output_shape, bs_masks=bs_masks) for _ in range(kfold)]
    if kfold == 1:
        return models[0]

    return ensemble(models)


if __name__ == '__main__':
    model = Model_1((72, 2, 256), 2)
    model.load_weights('modelSubmit_1.h5')
    model.save('modelSubmit_1.h5', include_optimizer=False)
