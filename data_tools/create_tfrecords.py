''' Creates tfrecords given GeoTIFF files.
We provide a copy of the dataset in tfrecords format.
You should not need this script unless you modify the dataset.
'''

import os, glob
import argparse
import random
import math
import tifffile as tiff
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _create_tfexample(B2, B3, B4, B5, B6, B7, label):
    example = tf.train.Example(features=tf.train.Features(feature={
            'B2': _bytes_feature(B2),
            'B3': _bytes_feature(B3),
            'B4': _bytes_feature(B4),
            'B5': _bytes_feature(B5),
            'B6': _bytes_feature(B6),
            'B7': _bytes_feature(B7),
            'L': _bytes_feature(label)
            }))
    return example

def preprocess_and_encode_sample(data_tensor):
    image = data_tensor[..., :-1]
    label = data_tensor[..., -1]

    image = tf.cast(image, tf.float32)
    image = image - tf.reduce_min(image)
    image = image / tf.maximum(tf.reduce_max(image), 1)
    image = image * 255

    image = tf.cast(image, tf.uint8)
    label = tf.cast(label, tf.uint8)

    B2 = tf.image.encode_png(image[..., 0, None])
    B3 = tf.image.encode_png(image[..., 1, None])
    B4 = tf.image.encode_png(image[..., 2, None])
    B5 = tf.image.encode_png(image[..., 3, None])
    B6 = tf.image.encode_png(image[..., 4, None])
    B7 = tf.image.encode_png(image[..., 5, None])
    L = tf.image.encode_png(label[..., None])
    return [B2, B3, B4, B5, B6, B7, L]

def create_tfrecords(save_dir, dataset_name, filenames, images_per_shard):
    data_placeholder = tf.placeholder(tf.uint16)
    processed_bands = preprocess_and_encode_sample(data_placeholder)

    with tf.Session() as sess:
        num_shards = math.ceil(len(filenames) / images_per_shard)
        for shard in range(num_shards):
            output_filename = os.path.join(save_dir, '{}_{:03d}-of-{:03d}.tfrecord'
                                        .format(dataset_name, shard, num_shards))
            print('Writing into {}'.format(output_filename))
            filenames_shard = filenames[shard*images_per_shard:(shard+1)*images_per_shard]

            with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                for filename in filenames_shard:
                    data = tiff.imread(filename)
                    B2, B3, B4, B5, B6, B7, L = sess.run(processed_bands, feed_dict={data_placeholder: data})
                    example = _create_tfexample(B2, B3, B4, B5, B6, B7, L)
                    tfrecord_writer.write(example.SerializeToString())

    print('Finished writing {} images into TFRecords'.format(len(filenames)))

def main(args):
    path = os.path.join(args.input_dir, '**/*.tif')
    filenames = glob.glob(path)

    random.seed(args.seed)
    random.shuffle(filenames)

    num_test = args.num_test_images

    # create TFRecords for the training and test sets
    create_tfrecords(save_dir=args.output_dir,
                     dataset_name='train',
                     filenames=filenames[num_test:],
                     images_per_shard=args.images_per_shard)
    create_tfrecords(save_dir=args.output_dir,
                     dataset_name='test',
                     filenames=filenames[:num_test],
                     images_per_shard=args.images_per_shard)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='E:/global_water_dataset',
                        help='path to the directory where the images will be read from')
    parser.add_argument('--output_dir', type=str, default='E:/tfrecords',
                        help='path to the directory where the TFRecords will be saved to')
    parser.add_argument('--images_per_shard', type=int, default=5000,
                        help='number of images per shard')
    parser.add_argument('--num_test_images', type=float, default=5000,
                        help='number of images in the test set')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for repeatable train/test splits')
    args = parser.parse_args()
    main(args)