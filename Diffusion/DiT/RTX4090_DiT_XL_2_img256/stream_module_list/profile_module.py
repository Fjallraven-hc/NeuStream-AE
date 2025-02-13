import copy
import time
from DiT_module import DiTModule
from vae_module import VaeModule
parameter_path = "/home/yhc/DiT/results/000-DiT-S-2/checkpoints/0350000.pt"
device = "cuda"
data_type = "data"
config = {}
DiT_module = DiTModule(device, data_type, parameter_path, config)
vae_module = VaeModule(device, data_type, "path")

DiT_module.deploy()
vae_module.deploy()

request_one = {
    "image_size": 256,
    "num-sampling-steps": 250,
    "timestep": {
        "loop_count": 249
    },
    "class_label": 207,
    "cfg_scale": 4.0
}

loop_count = 30 # 每个batch_size取30次平均

init_latency_list = []
for _ in range(loop_count):
    test_request = copy.deepcopy(request_one)
    begin = time.perf_counter()
    test_request = DiT_module.init_noise(test_request)
    test_request = DiT_module.init_betas_and_timesteps(test_request)
    end = time.perf_counter()
    init_latency_list.append(end - begin)

DiT_loop_count = 50
DiT_latency_list = {}
# test_request has been initialized
for batch_size in range(1, 41):
    print(f"profiling DiT_module batch_size = {batch_size}")
    DiT_latency_list[batch_size] = []
    for _ in range(DiT_loop_count):
        batch_request = [copy.deepcopy(test_request) for _ in range(batch_size)]
        begin = time.perf_counter()
        batch_request = DiT_module.compute(batch_request)
        end = time.perf_counter()
        DiT_latency_list[batch_size].append(end - begin)

DiT_latency = {}
for batch_size in range(1, 41):
    DiT_latency[batch_size] = sum(DiT_latency_list[batch_size]) / len(DiT_latency_list[batch_size])

test_request = batch_request[0]
vae_latency_list = {}
# test_request has been DiT_module calculated
for batch_size in range(1, 13):
    print(f"profiling vae_module batch_size = {batch_size}")
    vae_latency_list[batch_size] = []
    for _ in range(loop_count):
        batch_request = [copy.deepcopy(test_request) for _ in range(batch_size)]
        begin = time.perf_counter()
        batch_request = vae_module.compute(batch_request)
        end = time.perf_counter()
        vae_latency_list[batch_size].append(end - begin)

vae_latency = {}
for batch_size in range(1, 13):
    vae_latency[batch_size] = sum(vae_latency_list[batch_size]) / len(vae_latency_list[batch_size])
