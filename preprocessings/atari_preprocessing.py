from gym.core import ObservationWrapper
from gym.spaces import Box
import numpy as np
import cv2


class AtariPreprocess(ObservationWrapper):
    def __init__(self, env, img_size=(64, 64, 1), colors=False,
                 gray_scale_weights=None, crop_func=None):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)
        self.img_size = img_size
        self.colors = colors
        self.crop_func = crop_func if crop_func else lambda x: x[32:-16, :, :]
        self.gray_scale_weights = gray_scale_weights if gray_scale_weights else [0.7, 0.3, 0.2]
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb_image):
        return np.dot(rgb_image[..., :3], self.gray_scale_weights)

    def _resize(self, image):
        return cv2.resize(image, self.img_size[:2])

    def _scale(self, image):
        return (image / 255).astype('float32')

    def observation(self, img):
        """what happens to each observation"""
        img = self.crop_func(img)
        img = self._resize(img)
        if not self.colors:
            img = self._to_gray_scale(img)
        img = self._scale(img)
        img = img.reshape(self.img_size)
        return img

