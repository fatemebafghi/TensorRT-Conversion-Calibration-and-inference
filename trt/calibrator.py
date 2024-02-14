import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
from logzero import logger
import os


class Int8Calibrator(trt.IInt8EntropyCalibrator):
    def __init__(self, input_layers, stream):
        trt.IInt8EntropyCalibrator.__init__(self)
        self.input_layers = input_layers
        self.stream = stream
        self.device_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = "cache"
        self.stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names, p_str=None):
        # time.sleep(0.001)
        batch = self.stream.next_batch()
        # try:
        #     # Assume self.batches is a generator that provides batch data.
        #     # data = next(self.batches)
        #     # Assume that self.device_input is a device buffer allocated by the constructor.
        #     cuda.memcpy_htod(self.device_input, batch)
        #     return [int(self.device_input)]
        # except StopIteration:
        #     # When we're out of batches, we return either [] or None.
        #     # This signals to TensorRT that there is no calibration data remaining.
        #     return None
        # time.sleep(0.001)
        if batch.size == 0:
            logger.info("end of calibration data")
            return None

        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class ImageBatchStream:
    def __init__(self, batch_size, calibration_files, preprocessor, data_path):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                           (1 if (len(calibration_files) % batch_size)
                            else 0)
        # self.configs = configs
        self.files = calibration_files
        # self.calibration_data = np.zeros((batch_size, self.configs.channels, self.configs.width, self.configs.heigh),
        # dtype=np.float32)
        self.calibration_data = np.zeros((batch_size, 3, 192, 192),
                                         dtype=np.float32)
        self.batch = 0
        self.preprocessor = preprocessor
        self.data_path = data_path

    def read_image_chw(self, path):
        img = cv2.imread(os.path.join(self.data_path, path))
        return img

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch:
                                         self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                logger.info(f"[ImageBatchStream] Processing {f}")
                img = self.read_image_chw(f)
                img = self.preprocessor(img)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(
                self.calibration_data, dtype=np.float32)
        else:
            return np.array([])
