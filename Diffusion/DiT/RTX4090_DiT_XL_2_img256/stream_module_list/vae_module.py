import sys
import os
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

class VaeModule(StreamModule):
    def __init__(self, device, data_type, parameter_path, **kwargs):
        super().__init__(device=device)
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path

    def deploy(self, **kwargs):
        from diffusers.models import AutoencoderKL
        local_path = "../sd-vae-ft-mse"
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(local_path).to(self.device)
        if self.data_type == torch.float16:
            self.vae.half()
        self.deployed = True
    
    def offload(self, **kwargs):
        # offload model from GPU
        self.vae = self.vae.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False

    def compute(self, batch_request, **kwargs):
        if not self.deployed:
            raise CustomError("VaeModule is not deployed! Can not exec batch!")
        
        latents = []
        for request in batch_request:
            latents.append(request["latent"])
        latents = torch.stack(latents).to(self.device)

        samples = self.vae.decode(latents / 0.18215).sample
        # save image method, for reference
        # from torchvision.utils import save_image
        # save_image(samples[idx], "{name}.png", normalize=True)

        for idx in range(len(batch_request)):
            batch_request[idx]["image_numpy_ndarray"] = samples[idx].cpu().numpy()
            # save_image(samples[idx], f"{idx}.png", normalize=True) # not necessary
        
        return batch_request

        