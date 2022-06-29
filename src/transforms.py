import random
import torch
import numpy as np
from PIL import Image, ImageEnhance


class ColorJitter(object):
    r"""
    Apply color transform to the image, only one at a time.
    Args:
        color (tuple): (min, max) value for the color balance.
        brightness (tuple): (min, max) value for the brightness.
        contrast (tuple): (min, max) value for the color contrast.
        sharpness (tuple): (min, max) value for the color sharpness.
        p (float): Probability to apply rotation.
    """
    def __init__(self, color=(0., 2.), brightness=(1., 2.), contrast=(1., 2.), sharpness=(0., 2.), p=0.5):
        self.color = color
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image = sample.get('image')
            color_val = random.uniform(self.color[0], self.color[1])
            brightness_val = random.uniform(self.brightness[0], self.brightness[1])
            contrast_val = random.uniform(self.contrast[0], self.contrast[1])
            sharpness_val = random.uniform(self.sharpness[0], self.sharpness[1])
            options = [1, 2, 3, 4]
            random.shuffle(options)

            for option in options:
                if option == 1:
                    image = ImageEnhance.Color(image).enhance(color_val)
                elif option == 2:
                    image = ImageEnhance.Brightness(image).enhance(brightness_val)
                elif option == 3:
                    image = ImageEnhance.Contrast(image).enhance(contrast_val)
                else:
                    image = ImageEnhance.Sharpness(image).enhance(sharpness_val)

            sample['image'] = image
        return sample


class Rotate(object):
    r"""
    Apply rotation to the image.
    Args:
        max_angle (int): Max angle to rotate.
        p (float): Probability to apply rotation.
    """
    def __init__(self, max_angle=45, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, label, mask = sample.get('image'), sample.get('label'), sample.get('mask', None)
            angle = random.uniform(-self.max_angle, self.max_angle)
            image, label = image.rotate(angle, Image.BICUBIC), label.rotate(angle, Image.NEAREST)

            if mask is not None:
                mask = mask.rotate(angle, Image.NEAREST)

            sample = {'image': image, 'label': label, 'mask': mask}

        return sample


class ToTensor(object):
    r"""
    Convert data sample to pytorch tensor.
    """
    def __call__(self, sample):
        image, label, mask = sample.get('image'), sample.get('label'), sample.get('mask')

        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(label, Image.Image):
            label = np.array(label)

        image = torch.from_numpy(image.copy().transpose((2, 0, 1)).astype('float32') / 255.)
        label = torch.from_numpy(np.expand_dims(label, 0).astype('float32').copy())

        if mask is not None:
            mask = torch.from_numpy(mask.copy().astype('float32'))

        return {'image': image, 'label': label, 'mask': mask}
