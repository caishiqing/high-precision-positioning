from tensorflow.keras import layers
import tensorflow as tf


class SelfAttention(layers.MultiHeadAttention):
    def call(self, x, mask=None, training=None):
        return super(SelfAttention, self).call(
            x, x, attention_mask=mask, training=training)

    def compute_mask(self, x, mask=None):
        if mask is not None:
            return mask

        return tf.reduce_any(tf.not_equal(x, 0), axis=-1)


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

        cls_mask = tf.cast(tf.ones(shape=(tf.shape(mask)[0], 1)), mask.dtype)
        mask = tf.concat([cls_mask, mask], axis=1)
        return mask


def Residual(fn, res, dropout=0.0):
    x = fn(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([res, x])
    x = layers.LayerNormalization()(x)
    return x


def CIRNet(x):
    x = layers.TimeDistributed(layers.Conv1D(8, 3, padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.TimeDistributed(layers.Conv1D(16, 3, padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.TimeDistributed(layers.Conv1D(32, 3, padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPool1D(padding="same"))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    return x


def build_model(input_shape,
                output_shape=2,
                embed_dim=256,
                hidden_dim=512,
                num_heads=8,
                num_attention_layers=6,
                dropout=0.0):

    assert embed_dim % num_heads == 0
    dim_per_head = embed_dim // num_heads

    x = layers.Input(shape=input_shape)
    h = layers.Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]))(x)
    h = CIRNet(h)
    h = layers.Dense(embed_dim)(h)
    h = AntennaMasking()([x, h])
    h = AntennaEmbedding()(h)

    h = layers.Dropout(dropout)(h)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU()(h)

    for _ in range(num_attention_layers):
        h = Residual(SelfAttention(num_heads, dim_per_head, dropout=dropout), h)
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
    y = layers.Dense(output_shape)(h)

    model = tf.keras.Model(x, y)
    return model


tf.keras.utils.get_custom_objects().update(
    {
        'SelfAttention': SelfAttention,
        'AntennaMasking': AntennaMasking,
        'AntennaEmbedding': AntennaEmbedding
    }
)


if __name__ == '__main__':
    model = build_model((4, 2, 32), embed_dim=4, hidden_dim=8, num_heads=2)
    model.summary()

    import numpy as np
    x = np.random.random((1, 4, 2, 32))
    x[:, 2, :, :] = 0
    x = tf.constant(x)
    y = model(x)
    pass
