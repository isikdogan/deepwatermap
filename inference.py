import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import deepwatermap
import tifffile as tiff
import numpy as np
import cv2

model = deepwatermap.model()
model.load_weights('checkpoints_scratch/cp.007.ckpt')

def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

image = tiff.imread('test/p001r062_WC_20010731.tif')
pad_r = find_padding(image.shape[0])
pad_c = find_padding(image.shape[1])
image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')
#label = image[:,:, -1]
#image = image[..., :-1]
image = image.astype(np.float32)
image = image - np.min(image)
image = image / np.maximum(np.max(image), 1)

#mndwi = (image[..., 1] - image[..., 4]) / (image[..., 1] + image[..., 4] + 0.01)
#mndwi = mndwi / np.max(mndwi)
#cv2.imwrite('mndwi.png', mndwi * 255)

#cv2.imwrite('label.png', label * 85)

image = np.expand_dims(image, axis=0)
dwm = model.predict(image)
dwm = np.squeeze(dwm)
dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]
dwm[dwm < 0.5] = 0
#dwm[dwm >= 0.5] = 1

cv2.imwrite('dwm_7_p001r062_WC_20010731.png', dwm * 255)

'''
activation_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('activation_8').output)
dwm = activation_model.predict(image)
dwm = np.squeeze(dwm)
for i in range(4):
    f = dwm[:,:,i]
    f = f - np.min(f)
    f = f / (np.max(f) + 0.01)
    cv2.imwrite('f_{}.png'.format(i), f*255)
'''