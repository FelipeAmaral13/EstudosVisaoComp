import numpy as np
import cv2
from skimage import io, img_as_ubyte
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
import random
import os


class AugmentImage:

    def __init__(self, images_path, augmented_path, images_to_generate=10):
        self.images_path = images_path
        self.augmented_path = augmented_path
        self.images_to_generate = images_to_generate
        self.transformations = {
            'Rotacao anti-horaria': self.rotate_left,
            'Horizontal flip': self.h_flip,
            'Vertical flip': self.v_flip,
            'Rotacao horaria': self.rotate_right,
            'warp shift': self.warp_shift,
           'Ruidos': self.ruidos_img,
        #    'Brilho': self.brightness,
           'Blur Image': self.blur_img,
        }
        self.images = [os.path.join(images_path, im) for im in os.listdir(images_path)]

    def rotate_left(self, image):
        angle = random.randint(0, 180)
        return rotate(image, angle)

    def rotate_right(self, image):
        angle = random.randint(0, 180)
        return rotate(image, -angle)

    def h_flip(self, image):
        return np.fliplr(image)

    def v_flip(self, image):
        return np.flipud(image)

    def ruidos_img(self, image):
        return random_noise(image)

    # def brightness(self, image):
    #     bright = np.ones(image.shape, dtype="uint8") * 70
    #     brightincrease = cv2.add(image, bright)
    #     return brightincrease
    
    def warp_shift(self, image):
        transform = AffineTransform(translation=(0, 40))
        warp_image = warp(image, transform, mode="wrap")
        return warp_image

    def blur_img(self, image):
        k_size = random.randrange(1, 10, 2)
        img_blur = cv2.medianBlur(image, k_size)
        return img_blur

    def transform_image(self, image_path):
        image = io.imread(image_path)
        transformed_image = image
        transformation_count = random.randint(1, len(self.transformations))
        n = 0

        while n < transformation_count:
            key = random.choice(list(self.transformations))
            transformed_image = self.transformations[key](transformed_image)
            n += 1

        return transformed_image

    def generate_augmented_images(self):
        augmented_images = []
        for i in range(1, self.images_to_generate+1):
            image_path = random.choice(self.images)
            transformed_image = self.transform_image(image_path)
            new_image_path = os.path.join(self.augmented_path, f'augmented_image_{i}.jpg')
            io.imsave(new_image_path, img_as_ubyte(transformed_image))
            augmented_images.append(new_image_path)

        return augmented_images


augmenter = AugmentImage(images_path='Original', augmented_path='Augmented')

augmented_images = augmenter.generate_augmented_images()
