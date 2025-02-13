from typing import List
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from stream_module_list import UNetModule
import argparse

class Palette_pipe:
    def __init__(self, device):
        self.device = device
        parameter_path = "./16_Network.pth"
        self.unet = UNetModule(device=device, parameter_path=parameter_path)
        self.unet.deploy()

    def __call__(self, id_list: List[int], num_sampling_steps: int):        
        for _ in range(num_sampling_steps):
            self.unet.compute([-1 for _ in range(len(id_list))])
        return

