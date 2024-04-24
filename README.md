# TensorRT

## Project Overview

This project uses TensorRT to optimize and accelerate deep learning models for deployment on NVIDIA GPUs. The project
has three main parts: conversion, calibration, and inference. The conversion part is responsible for converting trained 
ONNX model inta format that can be used by TensorRT, which is called TensorRT engine. The calibration part is
responsible for calibrating the models to optimize performance INT8 precision. Finally, the inference part is 
responsible for performing inference on the optimized models using TensorRT.

## Installation

To use this project, you will need to install the following dependencies:

- NVIDIA GPU with CUDA support
- TensorRT
- pytorch
- Python 3.8

You can install TensorRT using NVIDIA's instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
You can install the required deep learning framework using their respective instructions.
you can also install it using python package installation by following [these section command](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)

## Configuration

Before using the project, you will need to configure it using the `config.yaml` file. The configuration file contains the following settings:

```
log_path: ./logs.log
paths:
  result_path: ./trt_files
  output_path: outputs
  log_file: logfile.log


model:
  onnx_file_path: "/model.onnx"
  input_min_size: (1, 3, 224, 224)
  input_opt_size: "change it according to your needs"
  input_max_size: "change it according to your needs"

calibrator:
  quantization_mode: int8 #fp32, fp16 or int8
  file_path: "your calibration data path"


```

The `log_path` setting specifies the path where the logs will be saved. The `paths` section contains settings for the
path where the output files will be saved.

The `model` section contains settings for the ONNX model file path and input dimensions. The `calibrator`
section contains settings for quantization mode and the path to the calibration data.


## Conversion
How to Run
To create the TensorRT engine for the specified model and configuration, run the create_engine.py script with the -
-config_path argument to specify the path to the configuration file. For example, to use the config.yaml 
file in the trt directory, run the following command:

```
python create_engine.py --config_path trt/config.yaml
```
This will load the configuration from the specified file, build the TensorRT engine for the ONNX model,
and save the engine to the trt_files directory as a serialized .engine file.

This code reads the configuration file at `trt/config.yaml`, builds the configuration object using `build_tree_config`, 
and then passes the configuration object to the `build_engine` function to convert the model.

## Calibration

The calibration part of the project is responsible for calibrating the models to optimize performance 
on the target hardware.
### Data Preparation for Calibration
In order to perform calibration for your TRT engine, you will need a set of calibration data that accurately represents 
the distribution of your input data. You can use any dataset that is representative of your input data, 
and you will typically want to use a small subset of your training data. All your data should be in one folder.

The calibration data should be in the same format as the input data used for inference,
and it's important to preprocess the calibration data in the same way as the input data to ensure that the calibration
is accurate.

### Calibration Process
Once you have prepared your calibration data, you can perform calibration in the TRT conversion process by using the
Int8Calibrator class provided in the calibrator.py module


## How to Run

To run inference on a set of input images using the generated TensorRT engine, 
run the `inference.py` script with the following command:

`````
python inference.py --trt_engine_path [path-to-trt-engine] --test_path [path-to-test-data]
`````

This will load the TensorRT engine from the specified path, perform inference on all the images in the specified directory,
and save the results to a log file. The `--trt_engine_path` argument specifies the path to the `.trt` file that was generated 
by the `create_engine.py` script, and the `--test_path` argument specifies the path to the directory containing the input images.

By default, the script will output the predicted label for each image and whether it was classified correctly or not. 
At the end of the inference process, the script will output the accuracy and the duration of the inference process in seconds.

you can also visit [TensorRT Conversion: Transforming Deep Learning Models for High-Speed Inference](https://medium.com/@fatemebfg/tensorrt-conversion-transforming-deep-learning-models-for-high-speed-inference-36548bdca46c) for more information
