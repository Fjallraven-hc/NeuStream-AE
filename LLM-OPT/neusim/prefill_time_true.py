import os
import sys
# sys.path.append(os.path.abspath('../'))

from vllm import LLM, SamplingParams
import argparse
# import torch
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

    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
    )

    parser.add_argument(
        '--num_dedup',
        type=int,
        default=3,
    )
    return parser.parse_args()




if __name__ == '__main__':

    args = _parse_args()

    model_name = args.model
    tp_size = args.tp

    sampling_params = SamplingParams(temperature=0,
                                        top_p=1,
                                        max_tokens=1)

    # device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    device = 'H100'
    directory = os.path.join(os.path.abspath(args.output_dir), device, model_name.replace('/', '_'))
    os.makedirs(directory, exist_ok=True)


    llm = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=0.95,max_num_batched_tokens=10000)
    try:
        llm.llm_engine.scheduler[0].do_predict = False
    except:
        print("No scheduler")

    # warmup
    np.random.seed(1033)
    input_ids = [np.random.randint(0, 30000, (2000,)).tolist()]
    for _ in range(2):
        llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    ####################
        

    llm.step_time_ls.clear()


    seq_lens = list(range(32, 2048+1, 32))
    satuate_len = 2048
    NUM_DEDUP = args.num_dedup

    filename = f"{directory}/true_prefill_{model_name.split('/')[-1]}_tp{tp_size}.log"

    f = open(filename, mode='w')

    print(f"seq len = {seq_lens}", file=f)
    for _ in range(NUM_DEDUP):
        for seq_len in seq_lens:
            max_bs = 1
            for bs in range(1, max_bs+1):
                np.random.seed(1033)
                input_ids = [np.random.randint(0, 30000, (seq_len,)).tolist() for _ in range(bs)]
                llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
        
        print(f"time_ls: {llm.step_time_ls}", file=f)
        llm.step_time_ls.clear()

    f.close()

