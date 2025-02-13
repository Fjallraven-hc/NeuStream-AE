from typing import List
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models, DiT_XL_2
import argparse

class DiT_pipe:
    def __init__(self, device):
        self.device = device
        self.latent_size = 32
        self.DiT = DiT_XL_2(input_size=32, num_classes=1000).to(device).half()
        vae_model_path = "../sd-vae-ft-mse"
        self.vae = AutoencoderKL.from_pretrained(vae_model_path).to(device).half()

    def __call__(self, class_labels: List[int], num_sampling_steps: int, cfg_scale: float):
        
        diffusion = create_diffusion(str(num_sampling_steps))
        
        n = len(class_labels)
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=self.device, dtype=torch.float16)
        y = torch.tensor(class_labels, device=self.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0) # 都没必要加这个0
        y_null = torch.tensor([1000] * n, device=self.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        samples = diffusion.p_sample_loop(
            self.DiT.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = self.vae.decode(samples / 0.18215).sample
        return samples

