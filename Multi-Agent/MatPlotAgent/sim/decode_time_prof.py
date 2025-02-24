import os

from vllm import LLM, SamplingParams
import argparse
import torch
import json
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
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    model_name = args.model
    tp_size = args.tp

    sampling_params = SamplingParams(temperature=0,
                                        top_p=1,
                                        max_tokens=500)
    

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    directory = os.path.join(os.path.abspath(args.output_dir), device, model_name.replace('/', '_'))
    os.makedirs(directory, exist_ok=True)

    llm = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=0.95,max_num_batched_tokens=10000)

    seqlen_config = {
        "4": 32,
    }

    fnme = model_name.replace('/', '_') + "_".join((key+"*"+str(val) for key, val in seqlen_config.items()))
    seq_len = []
    for k, v in seqlen_config.items():
        seq_len += [int(k)] * v


    np.random.seed(1033)
    input_ids = [np.random.randint(0, 30000, (seqlen,)).tolist() for seqlen in seq_len]

    ### warmup ###
    outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    llm.step_time_ls.clear()
    llm.llm_engine.step_tokens.clear()
    ##############


    outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    filename = f"{directory}/decode_{fnme}_tp{tp_size}.log"
    with open(file=filename, mode="w") as f:
        print(f"model = {model_name}, seq len = {seq_len}, tpsize = {tp_size}", file=f)
        f.write(json.dumps(llm.step_time_ls))
        f.write("\n")
        f.write(json.dumps(llm.llm_engine.step_tokens))
        

