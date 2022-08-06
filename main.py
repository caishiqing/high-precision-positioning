from sklearn.model_selection import train_test_split
from data import load_data, MaskBS
from optimizer import AdamWarmup
from model import build_model
import tensorflow as tf
import fire


def train(data_file, label_file, save_path,
          pretrained_path=None, **kwargs):

    batch_size = kwargs.pop('batch_size', 128)
    infer_batch_size = kwargs.pop('infer_batch_size', batch_size)
    epochs = kwargs.pop('epochs', 100)
    learning_rate = kwargs.pop('learning_rate', 1e-3)
    test_size = kwargs.pop('test_size', 0.2)

    x, y = load_data(data_file, label_file)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    autoturn = tf.data.AUTOTUNE
    augment = MaskBS(18, 4, 18)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).map(augment, num_parallel_calls=autoturn).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).map(augment, num_parallel_calls=autoturn).batch(infer_batch_size)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                    save_best_only=True,
                                                    mode='min',
                                                    monitor='val_loss')

    total_steps = len(x_train) // batch_size * epochs
    optimizer = AdamWarmup(warmup_steps=int(total_steps * 0.1),
                           decay_steps=total_steps-int(total_steps * 0.1),
                           initial_learning_rate=learning_rate)

    model = build_model(x.shape[1:], dropout=0.1)
    if pretrained_path is not None:
        model.load_weights(pretrained_path)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mae)
    model.summary()
    model.fit(x=train_dataset,
              epochs=kwargs.pop('epochs', 10),
              validation_data=valid_dataset,
              callbacks=[checkpoint])


if __name__ == '__main__':
    fire.Fire()
