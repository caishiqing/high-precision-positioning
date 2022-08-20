from tensorflow.keras import layers
import tensorflow as tf
import types


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


class AntennaMasking(layers.Layer):
    def __init__(self, **kwargs):
        super(AntennaMasking, self).__init__(**kwargs)
        self.flatten = layers.TimeDistributed(layers.Flatten())
        self.supports_masking = True

    def call(self, inputs):
        x, h = inputs
        mask = self.compute_mask(inputs)
        h *= tf.cast(mask[:, :, tf.newaxis], h.dtype)
        return h

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def compute_mask(self, inputs, mask=None):
        x = self.flatten(inputs[0])
        return tf.reduce_any(tf.not_equal(x, 0), axis=-1)


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


def Residual(fn, res, dropout=0.0):
    x = fn(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([res, x])
    return x


# def TimeReduction(x):
#     x = layers.Lambda(lambda x: x[:, :, :128, :])(x)
#     x = layers.TimeDistributed(layers.Conv1D(8, 3, padding="same"))(x)
#     x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
#     x = Residual(
#         tf.keras.Sequential(
#             layers=[
#                 layers.TimeDistributed(layers.Conv1D(8, 1, padding='same')),
#                 layers.TimeDistributed(layers.BatchNormalization()),
#                 layers.TimeDistributed(layers.LeakyReLU()),
#                 layers.TimeDistributed(layers.Conv1D(8, 3, padding='same'))
#             ]
#         ),
#         x
#     )
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)

#     x = layers.TimeDistributed(layers.Conv1D(16, 3, padding="same"))(x)
#     x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
#     x = Residual(
#         tf.keras.Sequential(
#             layers=[
#                 layers.TimeDistributed(layers.Conv1D(16, 1, padding='same')),
#                 layers.TimeDistributed(layers.BatchNormalization()),
#                 layers.TimeDistributed(layers.LeakyReLU()),
#                 layers.TimeDistributed(layers.Conv1D(16, 3, padding='same'))
#             ]
#         ),
#         x
#     )
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)

#     x = layers.TimeDistributed(layers.Conv1D(32, 3, padding="same"))(x)
#     x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
#     x = Residual(
#         tf.keras.Sequential(
#             layers=[
#                 layers.TimeDistributed(layers.Conv1D(32, 1, padding='same')),
#                 layers.TimeDistributed(layers.BatchNormalization()),
#                 layers.TimeDistributed(layers.LeakyReLU()),
#                 layers.TimeDistributed(layers.Conv1D(32, 3, padding='same'))
#             ]
#         ),
#         x
#     )
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)

#     x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
#     x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
#     x = Residual(
#         tf.keras.Sequential(
#             layers=[
#                 layers.TimeDistributed(layers.Conv1D(64, 1, padding='same')),
#                 layers.TimeDistributed(layers.BatchNormalization()),
#                 layers.TimeDistributed(layers.LeakyReLU()),
#                 layers.TimeDistributed(layers.Conv1D(64, 3, padding='same'))
#             ]
#         ),
#         x
#     )
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)

#     x = layers.TimeDistributed(layers.Flatten())(x)
#     return x


# class RecoverSignal(layers.Layer):
#     def call(self, x):
#         image, real = x[:, :, :, 0], x[:, :, :, 1]
#         amplitude = tf.math.sqrt(tf.pow(image, 2) + tf.pow(real, 2))
#         phase = tf.math.atan(tf.math.divide_no_nan(image, real))
#         x = tf.stack([amplitude, phase], axis=-1)
#         return x

#     def compute_output_shape(self, input_shape):
#         return input_shape


def TimeReduction(x):
    xs = []
    for xi, filters in zip(tf.split(x, 4, axis=-2), [128, 48, 24, 8]):
        xi = layers.TimeDistributed(layers.ZeroPadding1D(2))(xi)
        xi = layers.TimeDistributed(layers.Conv1D(filters, 64))(xi)
        xi = layers.TimeDistributed(layers.GlobalMaxPool1D())(xi)
        xs.append(xi)

    x = layers.Concatenate(-1)(xs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def build_model(input_shape,
                output_shape=2,
                embed_dim=256,
                hidden_dim=512,
                num_heads=8,
                num_attention_layers=6,
                dropout=0.0):

    assert embed_dim % num_heads == 0

    x = layers.Input(shape=input_shape)
    h = layers.Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]))(x)
    h = TimeReduction(h)
    h = layers.Dense(embed_dim)(h)
    h = layers.Dropout(dropout)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = AntennaMasking()([x, h])
    h = AntennaEmbedding()(h)
    h = layers.Dense(embed_dim)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)

    for _ in range(num_attention_layers):
        h = Residual(SelfAttention(num_heads, embed_dim, dropout=dropout), h)
        h = layers.LayerNormalization()(h)
        h = Residual(
            tf.keras.Sequential(
                layers=[
                    layers.Dense(hidden_dim, activation='relu'),
                    layers.Dense(embed_dim)
                ]
            ),
            h
        )
        h = layers.LayerNormalization()(h)

    h = layers.Lambda(lambda x: x[:, 0, :])(h)
    y = layers.Dense(output_shape)(h)

    model = tf.keras.Model(x, y)
    model.save = types.MethodType(save, model)
    return model


def save(cls, filepath, overwrite=True, **kwargs):
    """ save model without optimizer states """
    kwargs['include_optimizer'] = False
    tf.keras.models.save_model(cls, filepath,
                               overwrite=overwrite,
                               **kwargs)


tf.keras.utils.get_custom_objects().update(
    {
        'MultiHeadAttention': MultiHeadAttention,
        'SelfAttention': SelfAttention,
        'AntennaMasking': AntennaMasking,
        'AntennaEmbedding': AntennaEmbedding
    }
)


def Model_2(input_shape, output_shape):
    return build_model(input_shape, output_shape)


if __name__ == '__main__':
    model = Model_2((72, 2, 256), 2)
    model.load_weights('modelSubmit_2.h5')
    model.save('modelSubmit_2.h5')
    model = tf.keras.models.load_model('modelSubmit_2.h5')
