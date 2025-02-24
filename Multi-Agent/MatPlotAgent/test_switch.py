from vllm import LLM, SamplingParams
import os
import random
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import argparse
import time
import torch

mp = {
    "codellama": "/data/xwh/CodeLlama-7b-Instruct-hf",
    "llava": "/data/xwh/llava-1.5-7b-hf"
}

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
args = parser.parse_args()
model_type = args.type
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
llm = LLM(model=mp[model_type], max_model_len=4096, name=model_type)
prompts_id = [1] +[ random.randint(10,100) for _ in range(500)]

llm.llm_engine.scheduler[0].do_predict = False
pd_time = []
for _ in range(5):
    # torch.cuda.synchronize()
    param = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=2, ignore_eos=True)
    st = time.time()
    for _ in range(4):
        llm._add_request({"prompt_token_ids": prompts_id}, param)
        llm.llm_engine.step()
        llm.llm_engine.step()
    # torch.cuda.synchronize()
    pd_time.append(time.time()-st)

llm.llm_engine.scheduler[0].do_predict = False
p_time = []
for _ in range(5):
    # torch.cuda.synchronize()
    st = time.time()
    # for _ in range(4):
    param = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=2, ignore_eos=True)
    for _ in range(4):
        llm._add_request({"prompt_token_ids": prompts_id}, param)
    for _ in range(8):
        llm.llm_engine.step()
    # torch.cuda.synchronize()
    p_time.append(time.time()-st)

avg_pd = sum(pd_time[-3:])/3
avg_p = sum(p_time[-3:])/3
rate = round((avg_pd-avg_p)/avg_pd * 100, 2)
print(f"switch cost: {rate} %")
## ppppdddd

## pdpdpdpd

"""
2024-10-18 01:45:01,315 - Predictor - INFO - Predictor Imported.----------------------------------------
INFO 10-18 01:45:02 llm_engine.py:233] Initializing an LLM engine (v0.6.1) with config: model='/data/xwh/CodeLlama-7b-Instruct-hf', speculative_config=None, tokenizer='/data/xwh/CodeLlama-7b-Instruct-hf', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=/data/xwh/CodeLlama-7b-Instruct-hf, use_v2_block_manager=False, num_scheduler_steps=1, enable_prefix_caching=False, use_async_output_proc=True)
INFO 10-18 01:45:03 model_runner.py:997] Starting to load model /data/xwh/CodeLlama-7b-Instruct-hf...
Loading pt checkpoint shards:   0% Completed | 0/3 [00:00<?, ?it/s]
/home/xwh/new_vllm_docker/vllm_0_6/vllm/vllm/model_executor/model_loader/weight_utils.py:424: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(bin_file, map_location="cpu")
Loading pt checkpoint shards:  33% Completed | 1/3 [00:02<00:05,  2.67s/it]
Loading pt checkpoint shards:  67% Completed | 2/3 [00:06<00:03,  3.29s/it]
Loading pt checkpoint shards: 100% Completed | 3/3 [00:10<00:00,  3.58s/it]
Loading pt checkpoint shards: 100% Completed | 3/3 [00:10<00:00,  3.44s/it]

INFO 10-18 01:45:13 model_runner.py:1008] Loading model weights took 12.5562 GB
used memory: 12878 MB
Used memory after: 13274 MB, diff: 396 MB
memory overhead: 3.65%
INFO 10-18 01:45:14 gpu_executor.py:122] # GPU blocks: 1055, # CPU blocks: 512
INFO 10-18 01:45:16 model_runner.py:1309] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-18 01:45:16 model_runner.py:1313] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-18 01:45:27 model_runner.py:1428] Graph capturing finished in 11 secs.
p 1
d 1
INFO 10-18 01:45:27 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:27 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:27 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:27 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:27 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 01:45:28 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.000476837158203125 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 01:45:29 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:30 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:30 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 01:45:30 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
[0.2721993923187256, 0.25843358039855957, 0.25900936126708984, 0.26012325286865234, 0.2591896057128906] 0.25944073994954425
[0.259000301361084, 0.2589395046234131, 0.25917506217956543, 0.2585411071777344, 0.2589268684387207] 0.2588810125986735



2024-10-18 08:31:37,098 - Predictor - INFO - Predictor Imported.----------------------------------------
INFO 10-18 08:31:38 llm_engine.py:233] Initializing an LLM engine (v0.6.1) with config: model='/data/xwh/llava-1.5-7b-hf', speculative_config=None, tokenizer='/data/xwh/llava-1.5-7b-hf', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=/data/xwh/llava-1.5-7b-hf, use_v2_block_manager=False, num_scheduler_steps=1, enable_prefix_caching=False, use_async_output_proc=True)
INFO 10-18 08:31:39 model_runner.py:997] Starting to load model /data/xwh/llava-1.5-7b-hf...
/home/xwh/miniconda3/envs/vllm_0_6/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/home/xwh/miniconda3/envs/vllm_0_6/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
Loading safetensors checkpoint shards:   0% Completed | 0/3 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  33% Completed | 1/3 [00:00<00:00,  5.18it/s]
Loading safetensors checkpoint shards: 100% Completed | 3/3 [00:00<00:00, 15.02it/s]

INFO 10-18 08:31:43 model_runner.py:1008] Loading model weights took 13.1342 GB
used memory: 13492 MB
Used memory after: 14012 MB, diff: 520 MB
memory overhead: 4.79%
INFO 10-18 08:31:45 gpu_executor.py:122] # GPU blocks: 962, # CPU blocks: 512
INFO 10-18 08:31:46 model_runner.py:1309] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-18 08:31:46 model_runner.py:1313] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-18 08:31:57 model_runner.py:1428] Graph capturing finished in 11 secs.
p 1
d 1
INFO 10-18 08:31:59 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:31:59 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:31:59 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:31:59 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:31:59 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
d 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:00 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.000476837158203125 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:01 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:02 scheduler.py:953] sche time: 0.0002384185791015625 ms. ontime/timeout: 0/0
p 1
INFO 10-18 08:32:02 scheduler.py:953] sche time: 0.0 ms. ontime/timeout: 0/0
p 1
d 1
d 1
d 1
d 1
[0.2705516815185547, 0.2584075927734375, 0.25835251808166504, 0.25871944427490234, 0.2584221363067627] 0.2584980328877767
[0.2580835819244385, 0.2584245204925537, 0.2582435607910156, 0.25801897048950195, 0.2581510543823242] 0.25813786188761395
"""