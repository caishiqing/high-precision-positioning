from sklearn.model_selection import train_test_split
from data import load_data, MaskBS
from model import build_model
import tensorflow as tf
import fire


def train(data_file, label_file, **kwargs):
    augment = MaskBS(18, 4, 18)
    x, y = load_data(data_file, label_file)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=kwargs.pop('test_size', 0.2))

    batch_size = kwargs.pop('batch_size', 128)
    infer_batch_size = kwargs.pop('infer_batch_size', batch_size)

    autoturn = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).map(augment, num_parallel_calls=autoturn).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).map(augment, num_parallel_calls=autoturn).batch(infer_batch_size)

    save_path = kwargs.pop('save_path', 'model.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                    save_best_only=True,
                                                    mode='min',
                                                    monitor='val_loss')
    model = build_model((78, 2, 256), dropout=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(kwargs.pop('learning_rate', 1e-3)),
                  loss=tf.keras.losses.mse)
    model.fit(x=train_dataset,
              validation_data=valid_dataset,
              callbacks=[checkpoint])


if __name__ == '__main__':
    fire.Fire()
