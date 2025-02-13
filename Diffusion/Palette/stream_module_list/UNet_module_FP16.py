import sys
import os
import torch
import numpy as np
from typing import List, Dict, Union

# Add the parent directory to sys.path
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
sys.path.insert(0, script_dir)

# Add the grandparent directory to sys.path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils import *
import torch
torch.set_grad_enabled(False)

from .network import Network

class UNetModule(StreamModule):
    def __init__(self, device, parameter_path, **kwargs):
        super().__init__(device=device)
        self.data_type = torch.float16
        self.parameter_path = parameter_path
        self.loop_module = True

    def deploy(self, **kwargs):
        self.unet = Network().to(torch.float16).to(self.device)
        self.unet.load_state_dict(torch.load(self.parameter_path, map_location="cpu"), strict=False)
        self.unet.set_new_noise_schedule(phase='test', device=self.device)
        self.simulate_y_cond = torch.randn((11, 3, 256, 256), dtype=torch.float16).to(self.device)
        self.simulate_y_t = torch.randn((11, 3, 256, 256), dtype=torch.float16).to(self.device)
        self.simulate_y_0 = torch.randn((11, 3, 256, 256), dtype=torch.float16).to(self.device)
        self.simulate_mask = torch.randn((11, 3, 256, 256), dtype=torch.float16).to(self.device)
        self.deployed = True

    def offload(self, **kwargs):
        self.unet = self.unet.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False

    def compute(self, batch_request):
        # execute one step in restoration function
        batch_size  = len(batch_request)
        timesteps = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
        self.unet.p_sample(self.simulate_y_t[:batch_size], timesteps, y_cond=self.simulate_y_cond[:batch_size])
        self.simulate_y_0[:batch_size] * (1. - self.simulate_mask[:batch_size]) + self.simulate_mask[:batch_size] * self.simulate_y_t[:batch_size]
        return batch_request
    
