''' Runs inference on a given GeoTIFF image.
'''

# Uncomment this to run inference on CPU if your GPU runs out of memory
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import deepwatermap
import tifffile as tiff
import numpy as np
import cv2

def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

def main(checkpoint_path, image_path, save_path):
    # load the model
    model = deepwatermap.model()
    model.load_weights(checkpoint_path)

    # load and preprocess the input image
    image = tiff.imread(image_path)
    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])
    image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')
    image = image.astype(np.float32)
    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    # run inference
    image = np.expand_dims(image, axis=0)
    dwm = model.predict(image)
    dwm = np.squeeze(dwm)
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]
    dwm[dwm < 0.5] = 0

    # save the output water map
    cv2.imwrite(save_path, dwm * 255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/cp.150.ckpt',
                        help="Path to the dir where the checkpoints are stored")
    parser.add_argument('--image_path', type=str, help="Path to the input GeoTIFF image")
    parser.add_argument('--save_path', type=str, help="Path where the output map will be saved")
    args = parser.parse_args()
    main(args.checkpoint_path, args.image_path, args.save_path)
