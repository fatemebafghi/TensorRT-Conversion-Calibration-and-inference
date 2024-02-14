import argparse
import time
import tensorrt as trt
import numpy as np
import os
from logzero import logger
import pycuda.driver as cuda
import torch
from trt.utils import get_data
import pycuda.autoinit



class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            logger.info("a instance already exists")
        return cls._instances[cls]



class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding_name, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel(metaclass=SingletonMeta):

    def __init__(self, engine_path, max_batch_size=1, dtype=np.float16):

        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = self.engine.max_batch_size
        # self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self, batch_size):

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            binding_dims = self.engine.get_binding_shape(binding)
            if binding == 'input':
                size = trt.volume((batch_size, binding_dims[1], binding_dims[2], binding_dims[3]))
            elif binding == 'output':
                size = trt.volume((batch_size, binding_dims[1]))
            # size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            logger.info(f"binding = {binding} and dtype = {dtype}")
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem, binding))
            else:
                output_shape = self.engine.get_binding_shape(binding)
                dims = trt.Dims2(output_shape)
                output_shape = (batch_size, dims[1])
                outputs.append(HostDeviceMem(host_mem, device_mem, binding, output_shape))

        return inputs, outputs, bindings, stream

    def do_inference(self, input_image, bindings, inputs, outputs, stream):
        logger.debug("-------------------------> In inference method <-----------------------")
        inputs[0].host = np.ascontiguousarray(input_image, dtype=np.float32)

        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

        result = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        logger.info(f"execution was successfull = {result}")
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()

        outputs_dict = {}
        outputs_shape = {}
        for out in outputs:
            outputs_dict[out.binding_name] = np.reshape(out.host, out.shape)
            outputs_shape[out.binding_name] = out.shape

        return outputs_shape, outputs_dict

    def __call__(self, input_image):
        torch.cuda.empty_cache()
        batch_size = input_image.shape[0]

        self.context.set_binding_shape(self.engine.get_binding_index("input"), input_image.shape)
        inputs, outputs, bindings, stream = self.allocate_buffers(batch_size)
        start_time = time.time()

        outputs_shape, outputs_trt = self.do_inference(input_image=input_image,
                                                  bindings=bindings, inputs=inputs,
                                                  outputs=outputs, stream=stream)
        duration = time.time() - start_time
        logger.info(f"duration = {duration}")
        outputs_shape = outputs_shape.get('output')
        outputs_trt = outputs_trt.get('output')
        label = np.argmax(outputs_trt)
        logger.info(f"output_shape = {outputs_shape}, outputs_trt = {outputs_trt}")
        return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trt_engine_path', default='model.trt', type=str)
    parser.add_argument('--test_path', default='your test data path', type=str)
    args = parser.parse_args()

    model = TrtModel(args.trt_engine_path)

    # ====================================== Test your Model here ======================================
    test_path =args.test_path
    test_data = get_data(test_path)
    true_positive = 0
    len = 0
    start_time = time.time()
    for image, label in test_data:
        len += 1
        result = model(image)
        logger.info(f"result = {result} and gt = {label}")
        if result == label:
            true_positive += 1
        else:
            logger.error("this result was not a true positive")


    end_time = time.time()

    accuracy = true_positive/len
    logger.info(f"accuracy = {accuracy}")
    logger.info(f"duration = {end_time - start_time}")