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

from .DiT_models_FP16 import DiT, DiT_XL_2, DiT_S_2
# from DiT_models_FP16 import DiT, DiT_S_2

class DiTModule(StreamModule):
    def __init__(self, device, data_type, parameter_path, DiT_config: Dict, **kwargs):
        super().__init__(device=device)
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        self.DiT_config = DiT_config
        self.loop_module = True

    def deploy(self, **kwargs):
        #self.DiT = DiT_XL_2(input_size=32, num_classes=1000)
        #self.DiT.load_state_dict(torch.load(self.parameter_path, map_location='cpu'))
        self.DiT = DiT_S_2(input_size=32, num_classes=1000)
        # self.DiT.load_state_dict(torch.load(self.parameter_path, map_location='cpu'))
        self.DiT = self.DiT.to(self.device)
        self.DiT.eval()
        if self.data_type == torch.float16:
            self.DiT.half()
        # avoid tensor IO operation when compute
        self.y_null = torch.tensor([1000] * 100).to(self.device)
        self.deployed = True
    
    def offload(self, **kwargs):
        # offload model from GPU
        self.DiT = self.vae.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].half()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + torch.zeros(broadcast_shape, device=timesteps.device)
    
    def space_timesteps(self, num_timesteps, section_counts):
        """
        Create a list of timesteps to use from an original diffusion process,
        given the number of timesteps we want to take from equally-sized portions
        of the original process.
        For example, if there's 300 timesteps and the section counts are [10,15,20]
        then the first 100 timesteps are strided to be 10 timesteps, the second 100
        are strided to be 15 timesteps, and the final 100 are strided to be 20.
        If the stride is a string starting with "ddim", then the fixed striding
        from the DDIM paper is used, and only one section is allowed.
        :param num_timesteps: the number of diffusion steps in the original
                            process to divide up.
        :param section_counts: either a list of numbers, or a string containing
                            comma-separated numbers, indicating the step count
                            per section. As a special case, use "ddimN" where N
                            is a number of steps to use the striding from the
                            DDIM paper.
        :return: a set of diffusion steps from the original process to use.
        """
        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim") :])
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(
                    f"cannot create exactly {num_timesteps} steps with an integer stride"
                )
            section_counts = [int(x) for x in section_counts.split(",")]
        size_per = num_timesteps // len(section_counts)
        extra = num_timesteps % len(section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        return set(all_steps)

    def init_betas_and_timesteps(self, request):
        betas = np.linspace(0.0001, 0.02, 1000)
        use_timesteps = self.space_timesteps(1000, str(request["num-sampling-steps"]))
        # use_timesteps = space_timesteps(1000, str(request["num-sampling-steps"]))
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        betas = np.array(new_betas)
        # recalculate after new betas
        # 挺诡异的这地方...
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        ) if len(posterior_variance) > 1 else np.array([])
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        request["betas"] = torch.Tensor(betas).to(self.device).half()
        request["timestep_map"] = torch.tensor(timestep_map).to(self.device).half()
        request["posterior_log_variance_clipped"] = torch.Tensor(posterior_log_variance_clipped).to(self.device).half()
        request["posterior_variance"] = torch.Tensor(posterior_variance).to(self.device).half()
        request["sqrt_recip_alphas_cumprod"] = torch.Tensor(sqrt_recip_alphas_cumprod).to(self.device).half()
        request["sqrt_recipm1_alphas_cumprod"] = torch.Tensor(sqrt_recipm1_alphas_cumprod).to(self.device).half()
        request["posterior_mean_coef1"] = torch.Tensor(posterior_mean_coef1).to(self.device).half()
        request["posterior_mean_coef2"] = torch.Tensor(posterior_mean_coef2).to(self.device).half()
        return request
    
    def init_noise(self, request, seed=0):
        torch.manual_seed(seed)
        request["remain_loop_count"] = torch.tensor(request["remain_loop_count"]).to(self.device).unsqueeze(0)
        latent_size = request["image_size"] // 8
        request["latent"] = torch.randn(4, latent_size, latent_size, device=self.device, dtype=torch.float16)
        request["uncond_latent"] = request["latent"].clone()
        request["class_label"] = torch.tensor([request["class_label"]]).to(self.device)
        return request

    def compute(self, batch_request, **kwargs):
        if not self.deployed:
            raise CustomError("DiTModule is not deployed! Can not exec batch!")
        # judge whether a request is initialized
        for request in batch_request:
            if "latent" not in request:
                request = self.init_noise(request)
                request = self.init_betas_and_timesteps(request)

        B, C = len(batch_request), batch_request[0]["latent"].shape[0]
        latents = []
        uncond_latents = []
        mapped_time_steps = []
        timesteps = []
        class_labels = []
        cfg_scale_list = []
        for request in batch_request:
            latents.append(request["latent"])
            uncond_latents.append(request["uncond_latent"])
            timesteps.append(request["remain_loop_count"])
            mapped_time_steps.append(request["timestep_map"][request["remain_loop_count"]:request["remain_loop_count"]+1])
            class_labels.append(request["class_label"])
            cfg_scale_list.append(request["cfg_scale"])
            
        latents = torch.stack(latents+uncond_latents)
        mapped_timesteps = torch.cat(mapped_time_steps*2)#.to(self.device)
        timesteps = torch.cat(timesteps*2)#.to(self.device)
        #class_labels = torch.cat([torch.cat(class_labels), torch.tensor([1000] * B).to(self.device)])
        class_labels = torch.cat([torch.cat(class_labels), self.y_null[:B]])
#cfg_scale_list *= 2
        
        model_output = self.DiT.forward_with_varying_cfg(latents, mapped_timesteps, class_labels, cfg_scale_list)
        #print(model_output)
        #print("model_output")
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        ## 接下来的第一步，算对正确的betas和posterior_log_variance_clipped
        ## 再处理_extract_into_tensor

        ## if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
        #print(f"yhc debug:: model_output.shape={model_output.shape}")
        assert model_output.shape == (B * 2, C * 2, *latents.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        # 以下两行的broadcast方式需要改了，各自独立计算
        broadcast_shape = latents.shape[1:]
        min_log = []
        max_log = []
        for idx in range(B):
            min_log.append(torch.full(broadcast_shape, batch_request[idx]["posterior_log_variance_clipped"][timesteps[idx]].item(), dtype=torch.float16).to(self.device))
            max_log.append(torch.full(broadcast_shape, torch.log(batch_request[idx]["betas"])[timesteps[idx]].item(), dtype=torch.float16).to(self.device))
        min_log = torch.stack(min_log * 2)
        max_log = torch.stack(max_log * 2)
        #print(f"yhc debug:: min_log.shape={min_log.shape}")
        #print(f"yhc debug:: max_log.shape={max_log.shape}")
        #print(f"yhc debug:: model_output.shape={model_output.shape}")
        #print(f"yhc debug:: model_var_values.shape={model_var_values.shape}")
        #min_log = self._extract_into_tensor(self.posterior_log_variance_clipped, timesteps, latents.shape)
        #max_log = self._extract_into_tensor(np.log(self.betas), timesteps, latents.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        # _predict_xstart_from_eps
        assert model_output.shape == latents.shape
        left = []
        right = []
        for idx in range(B):
            left.append(torch.full(broadcast_shape, batch_request[idx]["sqrt_recip_alphas_cumprod"][timesteps[idx]].item(), dtype=torch.float16).to(self.device))
            right.append(torch.full(broadcast_shape, batch_request[idx]["sqrt_recipm1_alphas_cumprod"][timesteps[idx]].item(), dtype=torch.float16).to(self.device))
        pred_xstart = torch.stack(left*2) * latents - torch.stack(right*2) * model_output

        # q_posterior_mean_variance
        assert pred_xstart.shape == latents.shape
        left = []
        right = []
        for idx in range(B):
            left.append(torch.full(broadcast_shape, batch_request[idx]["posterior_mean_coef1"][timesteps[idx]].item(), dtype=torch.float16).to(self.device))
            right.append(torch.full(broadcast_shape, batch_request[idx]["posterior_mean_coef2"][timesteps[idx]].item(), dtype=torch.float16).to(self.device))

        model_mean = torch.stack(left*2) * pred_xstart + torch.stack(right*2) * latents
        #print(f"yhc debug:: model_mean= {model_mean}")

        # p_sample after p_mean_variance
        # torch.manual_seed(0)
        noise = torch.randn_like(latents)
        nonzero_mask = (
            (timesteps != 0).half().view(-1, *([1] * (len(latents.shape) - 1)))
        )
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        for idx in range(B):
            batch_request[idx]["latent"] = sample[idx]
            batch_request[idx]["uncond_latent"] = sample[idx+B]
        return batch_request
