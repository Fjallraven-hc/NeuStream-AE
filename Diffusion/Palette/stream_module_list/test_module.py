import time
import torch
from tqdm import tqdm

from UNet_module_FP16 import UNetModule

device = "cuda:0"
path = "/home/yhc/StreamFlowExperiments/Palette/Palette-FP16/16_Network.pth"

unet_module = UNetModule(device, path)
unet_module.deploy()

for batch_size in range(11, 12):
    begin = time.perf_counter()
    print(f"test batch size:{batch_size}")
    for _ in tqdm(range(1000)):
        _ = unet_module.exec_batch([1 for _ in range(batch_size)])
    end = time.perf_counter()
    print(f"batch size={batch_size}, avg step latency={(end - begin) / 1000}")
