#!/usr/bin/env python3
import importlib

mods = ["torch", "torchvision", "mmcv", "mmengine", "mmdet", "mmdet3d", "numpy", "cv2"]
for name in mods:
    mod = importlib.import_module(name)
    print("{}={}".format(name, getattr(mod, "__version__", "no_version")))

import torch
print("torch.cuda", torch.version.cuda)
print("cuda.available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda.device0", torch.cuda.get_device_name(0))
    print("cuda.capability0", torch.cuda.get_device_capability(0))

try:
    import mmcv._ext as ext
    print("mmcv._ext", ext.__file__)
except Exception as exc:
    print("mmcv._ext import failed", repr(exc))
    raise
