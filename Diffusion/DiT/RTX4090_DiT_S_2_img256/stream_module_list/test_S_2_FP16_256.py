import time
import copy
from tqdm import tqdm
from DiT_module_S_2_256_FP16 import DiTModule
from vae_module import VaeModule
parameter_path = "placeholder"
device = "cuda"
data_type = "float16"
config = {}
DiT_module = DiTModule(device, data_type, parameter_path, config)
vae_module = VaeModule(device, data_type, "path")

DiT_module.deploy()
vae_module.deploy()

meta_request = {
    "image_size": 256,
    "num-sampling-steps": 250,
    "remain_loop_count": 249,
    "remain_loop_count_no_cuda": 249,
    "class_label": 100,
    "cfg_scale": 4.0,
}

batch_request = [copy.deepcopy(meta_request)]
begin = time.perf_counter()
for _ in tqdm(range(250)):
    batch_request = DiT_module.compute(batch_request)
    for request in batch_request:
        request["remain_loop_count"] -= 1
        request["remain_loop_count_no_cuda"] -= 1

batch_request = vae_module.compute(batch_request)

from torchvision.utils import save_image

# samples = vae_module.vae.decode(batch_request[0]["latent"].unsqueeze(0) / 0.18215).sample
# batch_request[0]["image_numpy_ndarray"] = samples[0].cpu().numpy()
# save_image(samples, "fp16_256_label1000.png", normalize=True) # not necessary

####################################################################################################
# below test multi batch size data
####################################################################################################
XL_2_FP16_256_batch_latency_info = {}
for batch_size in range(1, 6):
    print(f"batch_size={batch_size}")
    batch_request = []
    XL_2_FP16_256_batch_latency_info[batch_size] = []
    for _ in range(batch_size):
        batch_request.append(copy.deepcopy(meta_request))
    for _ in range(250):
        begin = time.perf_counter()
        batch_request = DiT_module.compute(batch_request)
        for request in batch_request:
            request["remain_loop_count"] -= 1
            request["remain_loop_count_no_cuda"] -= 1
        end = time.perf_counter()
        XL_2_FP16_256_batch_latency_info[batch_size].append(end - begin)

import numpy as np
for batch_size in XL_2_FP16_256_batch_latency_info.keys():
    print(f"batch_size={batch_size}, max={max(XL_2_FP16_256_batch_latency_info[batch_size])}, min={min(XL_2_FP16_256_batch_latency_info[batch_size])}, avg={np.mean(XL_2_FP16_256_batch_latency_info[batch_size])}")