from gym.core import ObservationWrapper
from gym.spaces import Box
import numpy as np
import cv2


class BreakoutPreprocess(ObservationWrapper):
    def __init__(self, env, img_size=(64, 64, 1)):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = img_size
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb_image, weights=[0.7, 0.3, 0.2]):
        return np.dot(rgb_image[..., :3], weights)

    def _crop_irrelevant(self, image):
        return image[32:-16, :, :]

    def _resize(self, image):
        return cv2.resize(image, self.img_size[:2])

    def _scale(self, image):
        return (image / 255).astype('float32')

    def observation(self, img):
        """what happens to each observation"""
        img = self._crop_irrelevant(img)
        img = self._resize(img)
        img = self._to_gray_scale(img)
        img = self._scale(img)
        img = img.reshape(self.img_size)
        return img

class InvadersPreprocess(BreakoutPreprocess):
    def __init__(self, env, img_size=(64, 64, 1)):
        BreakoutPreprocess.__init__(self, env, img_size)

    def _crop_irrelevant(self, image):
        return image[20:-10, :, :]

class KungFuPreprocess(BreakoutPreprocess):
    def __init__(self, env, img_size=(64, 64, 3)):
        BreakoutPreprocess.__init__(self, env, img_size)

    def _crop_irrelevant(self, image):
        return image[60:-30, 5:, :]
