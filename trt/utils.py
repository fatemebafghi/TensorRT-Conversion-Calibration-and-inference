import numpy as np
from box import Box
import torch
from logzero import logger
import os
from PIL import Image
from preprocess import classifier_preprocess
import time
from typing import List
def build_tree_config(config):
    """
    get yaml config and create a box type dictionary in order to call it like config.train_settings.device
    :return: a box type dictionary
    """
    config = Box.from_yaml(str(config))
    return config

def get_data(data_test_path, original=False):
    classes = sorted(os.listdir(data_test_path))
    image_batch = []
    for item in classes:
        class_path = os.path.join(data_test_path, item)
        for images in os.listdir(class_path):
            image_path = os.path.join(class_path, images)
            image = Image.open(image_path).convert("RGB")

            image = classifier_preprocess(image)
            image = np.expand_dims(image, axis=0)
            image = image.astype('f')
            yield image, classes.index(item)