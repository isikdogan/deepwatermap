''' Runs a georeferenced inference on a given GeoTIFF image.

example:
$ python geoinference.py --checkpoint_path checkpoints/cp.135.ckpt \
    --image_path sample_data/sentinel2_example.tif --save_path water_map.tif
'''

# Uncomment this to run inference on CPU if your GPU runs out of memory
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import deepwatermap
import tifffile as tiff
import numpy as np
import cv2
import rasterio as rio

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
    src = rio.open(image_path)
    image = src.read()
    image = np.moveaxis(image, [0,1,2], [2,0,1])
    
    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])
    image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')

    # solve no-pad index issue after inference
    if pad_r[1] == 0:
        pad_r = (pad_r[0], 1)
    if pad_c[1] == 0:
        pad_c = (pad_c[0], 1)

    image = image.astype(np.float32)

    # remove nans (and infinity) - replace with 0s
    image = np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    # run inference
    image = np.expand_dims(image, axis=0)
    dwm = model.predict(image)
    dwm = np.squeeze(dwm)
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

    # soft threshold
    dwm = 1./(1+np.exp(-(16*(dwm-0.5))))
    dwm = np.clip(dwm, 0, 1)
    
    dwm = np.expand_dims(dwm*255, axis=0)
    
    # get image metadata
    metadata = src.meta().copy()
    metadata.update({
    'dtype': img.dtype,
    'count': img.shape[0],
    'compress': 'lzw'})

    # save the output water map
    with rio.open(save_path, 'w', **metadata) as dst:
        dst.write(dwm)

    # if thumbnails==True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        help="Path to the dir where the checkpoints are stored")
    parser.add_argument('--image_path', type=str, help="Path to the input GeoTIFF image")
    parser.add_argument('--save_path', type=str, help="Path where the output map will be saved")
    args = parser.parse_args()
    main(args.checkpoint_path, args.image_path, args.save_path)

