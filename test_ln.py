import ctypes
import tensorrt as trt
import torch

soFile = './layernorm_plugin.so'
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
ctypes.cdll.LoadLibrary(soFile)

from fastldm.modules import NewLayerNorm

ln = NewLayerNorm(256)
x = torch.randn(2, 64, 256, 256)
with torch.no_grad():
    out_th = ln(x)
from torch.onnx import OperatorExportTypes
torch.onnx.export(ln, (x,), 'ln.onnx', operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_0'], output_names=['output_0'])

import os
os.system('trtexec --onnx=ln.onnx --saveEngine=ln.trt --buildOnly --plugins=/root/host/speedup/tensorrt-layernorm-plugin/layernorm_plugin.so')

import numpy as np
from polygraphy.backend.trt import TrtRunner
def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine('ln.trt')
feed_dict = {'input_0': x.numpy()}
with TrtRunner(engine) as runner:
    outputs_trt = runner.infer(feed_dict)

print('max error', np.abs(out_th.numpy()-outputs_trt['output_0']).max())
breakpoint()
