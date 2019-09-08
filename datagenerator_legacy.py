import numpy as np
import random
from pathlib import Path
import tifffile as tiff
import cv2

class DataGenerator:
    def __init__(self, image_size, base_dir, is_training):
        self.image_size = image_size
        self.base_dir = Path(base_dir)
        self.filenames = list(self.base_dir.glob('**/*.tif'))

        random.seed(42)
        random.shuffle(self.filenames)

        self.is_training = is_training
        if self.is_training:
            self.filenames = self.filenames[5000:]
        else:
            self.filenames = self.filenames[:5000]

        self.dataset_size = len(self.filenames)
        self.iter_number = 0

    def __iter__(self):
        return self

    def get_tensor_shape(self):
        data_shape_in = (self.image_size[0], self.image_size[1], 6)
        data_shape_out = (self.image_size[0], self.image_size[1], 1)
        return data_shape_in, data_shape_out

    def _crop_center(self, img):
        crop_h, crop_w = self.image_size
        pad_h = img.shape[0] // 2 - (crop_h//2)
        pad_w = img.shape[1] // 2 - (crop_w//2)
        return img[pad_h:pad_h+crop_h, pad_w:pad_w+crop_w]

    def __next__(self):
        # repeat when iterator runs out of data
        if self.iter_number >= self.dataset_size:
            self.iter_number = 0
            print("Epoch complete, data generator is reset.")

        # load next image
        image_path = self.filenames[self.iter_number]
        image = tiff.imread(str(image_path))
        image_in = image[..., :-1]
        image_out = image[..., -1, np.newaxis]

        if self.is_training:
            image_in, image_out = DataAugmenter.random_crop(image_in, image_out)

        # convert to float
        image_in = image_in.astype(np.float32)
        image_out = image_out.astype(np.float32)

        if self.is_training:
            image_in = DataAugmenter.random_distort(image_in)
            image_in, image_out = DataAugmenter.random_flip([image_in, image_out])

        # center crop images
        image_in = self._crop_center(image_in)
        image_out = self._crop_center(image_out)

        # normalize images
        image_in = image_in - np.min(image_in)
        image_in = image_in / np.maximum(np.max(image_in), 1)
        image_out = image_out / 3

        self.iter_number += 1
        return image_in, image_out

class DataAugmenter:
    @staticmethod
    def random_flip(images):
        if random.random() < 0.5:
            for i in range(len(images)):
                images[i] = images[i][:,::-1,...]
        return images

    @staticmethod
    def random_crop(image_in, image_out, crop_ratio=0.8):
        assert image_in.shape[0] == image_out.shape[0]
        assert image_in.shape[1] == image_out.shape[1]
        min_dim = np.minimum(image_in.shape[0], image_in.shape[1])
        min_dim = int(min_dim * crop_ratio)
        x = random.randint(0, image_in.shape[1] - min_dim)
        y = random.randint(0, image_in.shape[0] - min_dim)
        image_in = image_in[y:y+min_dim, x:x+min_dim]
        image_out = image_out[y:y+min_dim, x:x+min_dim]
        return image_in, image_out

    @staticmethod
    def random_distort(image_in):
        # types of image distortions to pick from for data augmentation
        dist_functions = [DataAugmenter.random_gaussian_blur,
                          DataAugmenter.random_noise,
                          DataAugmenter.random_channel_mixing]
        num_dist = random.randrange(1, len(dist_functions) + 1)
        random.shuffle(dist_functions)
        for i in range(num_dist):
            dist_func = dist_functions[i]
            image_in = dist_func(image_in)
        image_in = np.maximum(image_in, 0)
        return image_in

    @staticmethod
    def random_channel_mixing(image_in):
        channel_bleeding_sigma = np.random.random() * 0.6
        ccm = np.identity(6)
        ccm = cv2.GaussianBlur(ccm, (3, 3), channel_bleeding_sigma)
        image_in = image_in = np.dot(image_in, ccm)
        return image_in

    @staticmethod
    def random_gaussian_blur(image_in):
        k_size = random.randrange(3, 7, 2)
        image_in = cv2.GaussianBlur(image_in, (k_size, k_size), 0)
        return image_in

    @staticmethod
    def random_noise(image_in):
        noise_var = random.randrange(3, 7)
        h, w, c = image_in.shape
        noise = np.random.randn(h, w, c) * noise_var
        image_in += noise
        return image_in

if __name__ == '__main__':
    image_size = (512, 512)
    dg = DataGenerator(image_size, 'E:/global_water_dataset', is_training=True)
    image_in, label = next(dg)
    #for band in range(image_in.shape[-1]):
    #    cv2.imwrite('image_in_band{}.png'.format(band), image_in[:,:,band] * 255)
    #cv2.imwrite('label.png', label * 255)

    mndwi = (image_in[:,:,0] - image_in[:,:,4]) / (image_in[:,:,0] + image_in[:,:,4])
    mndwi = mndwi / np.max(mndwi)
    cv2.imwrite('mndwi.png', mndwi * 255)
