from vllm import LLM, SamplingParams
import os
import random
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/data/xwh/llava-1.5-7b-hf')# CodeLlama-7b-Instruct-hf')
parser.add_argument("--no_prefill", action='store_true')
parser.add_argument("--no_decode",  action='store_true')
parser.add_argument("--no_ex", action="store_true")
parser.add_argument('--type', type=str)
args = parser.parse_args()
model_type = args.type
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
llm = LLM(model=args.model, max_model_len=4096, name=model_type)
llm.llm_engine.scheduler[0].do_predict = False

prefill = not args.no_prefill
decode = not args.no_decode
model_name = args.model.split("/")[-1]

fname = f'/home/xwh/new_vllm_docker/profile_data/{model_type}.txt'
f = open(fname, 'a+')

if not args.no_ex:
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1, ignore_eos=True)
    unit_len = 32
    time_prefill = []
    for i in tqdm(range(1, 129)): # 
        tmp = []
        if i <= 20:
            num = 80
        else:
            num = 3
        for _ in range(num):
            prompts_id = [ random.randint(1,100) for _ in range(unit_len*i)]
            outputs = llm.generate(prompt_token_ids=[prompts_id], sampling_params=sampling_params, use_tqdm=False)
            # tmp.append(outputs[0].metrics.time_summary[0])
            assert len(outputs[0].step_time) == 1, f"Length Error.{len(outputs[0].step_time)}"
            tmp.append(outputs[0].step_time[0] - outputs[0].arrival_time)
        time_prefill.append(tmp)

    time_clean = [ sum(tt[-3:])/3 for tt in time_prefill]
    print(f"prefill time: {time_clean}", file=f)

## prefill
if prefill:
    length = [ 32*i for i in range(1,4096 // 32 + 1) ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1, ignore_eos=True)
    batch_size = [2**i for i in range(4)]
    res = {}

    for l in tqdm(length):
        for b in batch_size:
            if b*l > 4096 or b*l < 120:
                continue
            s = str(l)+"_"+str(b)
            res[s] = []
            for _ in range(20):
                prompts_id = [ random.randint(10,100) for _ in range(l)]
                outputs = llm.generate(prompt_token_ids=[prompts_id]*b, sampling_params=sampling_params, use_tqdm=False)
                res[s].append(outputs[0].step_time[0] - outputs[0].arrival_time)
    X = []
    y = []

    for k,v in res.items():
        l,b = k.split("_")
        l = int(l)
        b = int(b)
        X.append((b*l, b*l*l))
        y.append(sum(v)/len(v))

    model = LinearRegression()

    model.fit(X, y)

    a, b = model.coef_
    c = model.intercept_

    y_pred = model.predict(X)
    error = sum([(y[i]-y_pred[i])**2 for i in range(len(y))])

    print(f"prefill param: ({c}, {a}, {b})", file=f)
    print(f"prefill error: {error}", file=f)


## decode
if decode:
    length = 16
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=4000, ignore_eos=True)
    batch_size = [2**i for i in range(3)]
    res = {}

    for b in batch_size:
        prompts_id = [ random.randint(10,100) for _ in range(length)]
        outputs = llm.generate(prompt_token_ids=[prompts_id]*b, sampling_params=sampling_params, use_tqdm=False)
        for i,t in enumerate(outputs[0].step_time[1:]):
            ll = str(length+i) + "_" + str(b)
            res[ll] = t - outputs[0].step_time[i]

    X = []
    y = []

    re_start = defaultdict(list)

    for k,v in res.items():
        l,b = k.split("_")
        l = int(l)
        b = int(b)
        if b*l > 12500:
            continue
        re_start[b*l].append(v)

    for k,v in re_start.items():
        X.append((k,))
        y.append(sum(v) / len(v))

    model = LinearRegression()

    model.fit(X, y)

    a = model.coef_
    c = model.intercept_

    y_pred = model.predict(X)
    error = sum([(y[i]-y_pred[i])**2 for i in range(len(y))])
    print(f"decode param: ({c}, {a[0]})",file=f)
    print(f"decode error: {error}", file=f)


f.close()