''' Runs inference on a given GeoTIFF image.

example:
$ python inference.py --checkpoint_path checkpoints/cp.135.ckpt --image_path sample_data/image2.tif --save_path water_map.png
'''

# Uncomment this to run inference on CPU if your GPU runs out of memory
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from deepwatermap import deepwatermap
import tifffile as tiff
import numpy as np
import cv2

def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        help="Path to the dir where the checkpoints are stored")
    parser.add_argument('--image_path', type=str, help="Path to the input GeoTIFF image")
    parser.add_argument('--save_path', type=str, help="Path where the output map will be saved")
    args = vars(parser.parse_args())
    
    checkpoint_path=args["checkpoint_path"]
    image_path=args["image_path"]
    save_path=args["save_path"]    

    # load the model
    model = deepwatermap.model()
    model.load_weights(checkpoint_path)

    # load and preprocess the input image
    image = tiff.imread(image_path)
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

    # save the output water map
    cv2.imwrite(save_path, dwm * 255)

if __name__ == '__main__':    
    main()
