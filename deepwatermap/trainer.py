''' Trains a DeepWaterMap model. We provide a copy of the trained checkpoints.
You should not need this script unless you want to re-train the model.
'''

import os, glob
import argparse
import tensorflow as tf
import deepwatermap
from metrics import running_precision, running_recall, running_f1
from metrics import adaptive_maxpool_loss

class TFModelTrainer:
    def __init__(self, checkpoint_dir, data_path):
        self.checkpoint_dir = checkpoint_dir

        # set training parameters
        self.image_size = (512, 512)
        self.learning_rate = 0.1
        self.num_epoch = 150
        self.batch_size = 24

        # create the data generators
        train_filenames = glob.glob(os.path.join(data_path, 'train_*.tfrecord'))
        val_filenames = glob.glob(os.path.join(data_path, 'test_*.tfrecord'))

        self.dataset_train = self._data_layer(train_filenames)
        self.dataset_val = self._data_layer(val_filenames)

        self.dataset_train_size = 137682
        self.dataset_val_size = 5000
        self.steps_per_epoch = self.dataset_train_size // self.batch_size
        self.validation_steps = self.dataset_val_size  // self.batch_size

    def _data_layer(self, filenames, num_threads=24):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=num_threads)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=4)
        return dataset

    def _parse_tfrecord(self, example_proto):
        keys_to_features = {'B2': tf.io.FixedLenFeature([], tf.string),
                            'B3': tf.io.FixedLenFeature([], tf.string),
                            'B4': tf.io.FixedLenFeature([], tf.string),
                            'B5': tf.io.FixedLenFeature([], tf.string),
                            'B6': tf.io.FixedLenFeature([], tf.string),
                            'B7': tf.io.FixedLenFeature([], tf.string),
                            'L': tf.io.FixedLenFeature([], tf.string)}
        F = tf.io.parse_single_example(example_proto, keys_to_features)
        data = F['B2'], F['B3'], F['B4'], F['B5'], F['B6'], F['B7'], F['L']
        image, label = self._decode_images(data)
        return image, label

    def _decode_images(self, data_strings):
        bands = [[]] * len(data_strings)
        for i in range(len(data_strings)):
            bands[i] = tf.image.decode_png(data_strings[i])
        data = tf.concat(bands, -1)
        data = tf.image.random_crop(data, size=[self.image_size[0], self.image_size[1], len(data_strings)])
        data = tf.cast(data, tf.float32)
        image = data[..., :-1] / 255
        label = data[..., -1, None] / 3
        self._preprocess_images(image)
        return image, label

    def _preprocess_images(self, image):
        image = self._random_channel_mixing(image)
        image = self._gaussian_noise(image)
        image = self._normalize_image(image)
        return image

    def _random_channel_mixing(self, image):
        ccm = tf.eye(6)[None, :, :, None]
        r = tf.random.uniform([3], maxval=0.25) + [0, 1, 0]
        filter = r[None, :, None, None]
        ccm = tf.nn.depthwise_conv2d(ccm, filter, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        ccm = tf.squeeze(ccm)
        image = tf.tensordot(image, ccm, (-1, 0))
        return image

    def _gaussian_noise(self, image):
        r = tf.random.uniform((), maxval=0.04)
        image = image + tf.random.normal([self.image_size[0], self.image_size[1], 6], stddev=r)
        return image

    def _normalize_image(self, image):
        image = tf.cast(image, tf.float32)
        image = image - tf.reduce_min(image)
        image = image / tf.maximum(tf.reduce_max(image), 1)
        return image

    def _optimizer(self):
        optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=0.9)
        return optimizer

    def train(self):
        # Callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_dir, 'cp.{epoch:03d}.ckpt'),
                                                         save_weights_only=True)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        # Model
        model = deepwatermap.model()

        initial_epoch = 0
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model.load_weights(ckpt.model_checkpoint_path)
            print("Loaded weights from", ckpt.model_checkpoint_path)
            initial_epoch = int(ckpt.model_checkpoint_path.split('.')[-2])

        model.compile(optimizer=self._optimizer(),
                      loss=adaptive_maxpool_loss,
                      metrics=[tf.keras.metrics.binary_accuracy,
                               running_precision, running_recall, running_f1])
        model.fit(self.dataset_train,
                  validation_data=self.dataset_val,
                  epochs=self.num_epoch,
                  initial_epoch=initial_epoch,
                  steps_per_epoch=self.steps_per_epoch,
                  validation_steps=self.validation_steps,
                  callbacks=[cp_callback, tb_callback, lr_callback])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--data_path', type=str,
                        help="Path to the tfrecord files")
    args = parser.parse_args()
    trainer = TFModelTrainer(args.checkpoint_path, args.data_path)
    trainer.train()

if __name__ == '__main__':
    main()
