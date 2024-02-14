import numpy as np
import cv2


def classifier_preprocess(image):
    # image = Image.fromarray(image)
    image = SquarePad2()(image)
    image = image * (1 / 255)
    image = cv2.resize(image, dsize=(192, 192),
                       interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 1, 0))
    image[0] = (image[0] - 0.485) / 0.299
    image[1] = (image[1] - 0.456) / 0.224
    image[2] = (image[2] - 0.406) / 0.225
    image = np.expand_dims(image, axis=0)
    image = image.astype('f')
    return image


class SquarePad2:
    def __call__(self, image):
        _, w, h = image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        return cv2.copyMakeBorder(image, hp, vp, hp, vp, cv2.BORDER_CONSTANT, value=(0, 0, 0))
