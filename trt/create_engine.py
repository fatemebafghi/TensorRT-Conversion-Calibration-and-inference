import tensorrt as trt
# import tensorrt.tensorrt.DeviceType as DeviceType
from calibrator import ImageBatchStream, Int8Calibrator
from tensorrt import IInt8EntropyCalibrator
from logzero import logger
from preprocess import classifier_preprocess
import yaml
from utils import build_tree_config
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(configs, model_path, data_path=None, quantize="int8"):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = 1 << 50
        # builder.max_workspace_size = 1 << 20
        builder.max_batch_size = 1

        if quantize == "int8":
            calibration_files = os.listdir(data_path)
            batchstream = ImageBatchStream(2, calibration_files, classifier_preprocess, data_path)
            logger.info("preparing for calibration...")
            logger.debug("---------------------------------- before calibration ----------------------------------")
            int8_calibrator = Int8Calibrator(["input"], batchstream)
            # int8_calibrator = IInt8EntropyCalibrator(["input"], batchstream)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = int8_calibrator
            logger.debug("*********************** after calibration ************************")
        elif quantize == "fp16":
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.default_device_type = trt.DeviceType.GPU
        # config.set_tactic_sources(
        #     tactic_sources=1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)
        # )

        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
            parser.parse(f.read())

        logger.info("ONNX parse ended")
        profile = builder.create_optimization_profile()

        network.add_input("input", trt.float32, (-1, -1, -1, -1))
        profile.set_shape("input", configs.model.input_min_size, configs.model.input_opt_size, configs.model.input_max_size)
        config.add_optimization_profile(profile)


        logger.debug(f"config = {config}")


        logger.info("====================== building tensorrt engine... ======================")
        engine = builder.build_serialized_network(network, config)
        logger.info("engine was created successfully")
        trt_path = os.path.join(configs.path.trt_files, f'model_{quantize}.trt')
        with open(trt_path, 'wb') as f:
            try:
                f.write(bytearray(engine))
            except:
                logger.error("file not writen")
        return engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='trt/config.yaml',
                        help='path to configuration file')
    args = parser.parse_args()

    with open(args.config_path) as file:
        config_file = yaml.full_load(file)

    configs = build_tree_config(config_file)

    build_engine(configs,
                 model_path=configs.model.onnx_file_path,
                 data_path=configs.calibrator.file_path,
                 quantize=configs.calibrator.quantization_mode)
