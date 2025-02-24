# 用于测定decode阶段的batch latency 曲线

import os
import sys
sys.path.append(os.path.abspath('../'))

from vllm import LLM, SamplingParams
import argparse
import torch
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        type=str,
    )

    parser.add_argument(
        '--tp',
        required=False,
        type=int,
        default=1
    )
    return parser.parse_args()




if __name__ == '__main__':

    args = _parse_args()

    model_name = args.model
    tp_size = args.tp

    sampling_params = SamplingParams(temperature=0,
                                        top_p=1,
                                        max_tokens=3)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    directory = f"./data/{device}/{model_name.replace('/', '_')}"
    os.makedirs(directory, exist_ok=True)


    llm = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=0.95,max_num_batched_tokens=10000)
    llm.llm_engine.scheduler.do_predict = False

    # warmup
    np.random.seed(1033)
    input_ids = [np.random.randint(0, 30000, (2000,)).tolist()]
    for _ in range(2):
        llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    ####################
        

    llm.step_time_ls.clear()


    seq_lens = [128]
    satuate_len = 1024
    NUM_DEDUP = 1

    filename = f"{directory}/prefill_{model_name.split('/')[-1]}_tp{tp_size}_bslatency.log"
    f = open(filename, mode='w')

    ls1, ls2 = [], []
    bs_ls = [1, 2, 4, 8, 16, 32, 64, 128]

    for seq_len in seq_lens:
        # max_bs = int(satuate_len/seq_len)
        # print(f"seq len = {seq_len}\nbatch_size: {list(range(1, max_bs+1))}", file=f)
        for _ in range(NUM_DEDUP):
            for bs in bs_ls:
                np.random.seed(1033)
                input_ids = [np.random.randint(0, 30000, (seq_len,)).tolist() for _ in range(bs)]
                llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
                ls1.append(llm.step_time_ls[-2])
                ls2.append(llm.step_time_ls[-1])
                llm.step_time_ls.clear()

    print(f"bs: {bs_ls}\ntime1: {ls1}\ntime2: {ls2}", file=f)
    f.close()

