import numpy as np
import multiprocessing as mp
import time
import logging
import os
import json
import glob
import shutil
from copy import deepcopy
import random
from neustream.code_executor import CodeExecutor
from vllm import LLM, SamplingParams
from vllm.logger import add_consolehandler, add_filehandler
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--chunk',action="store_true")

args = parser.parse_args()


workspace = "./workspace3"
GPU_MAP = {0:"4",2:"5", 3:"6",4:"7"}

cv_seed = {1: 4, 2: 6, 3: 8, 4: 6}
DUMP = 3

ori_test = [
    # rate 
    (0.6, 1, 1.8), (0.7, 1, 1.8), (0.8, 1, 1.8),(0.9, 1, 1.8), (1.0, 1, 1.8),
    # cv
    (0.7,1,1.8), (0.7, 2, 1.8), (0.7, 3, 1.8), (0.7, 4, 1.8),
    # slo
    (0.7, 1, 1.2), (0.7, 1, 1.4), (0.7, 1, 1.6), (0.7, 1, 1.8)
] * DUMP



test = []
for index, tt in enumerate(ori_test):
    # if index != 0:
    test.append(tuple(tt+(True, )))
    test.append(tuple(tt+(False,)))

loop = 300
test_num = len(test)

def frontend(input_queue: mp.Queue,output_queue: list[mp.Queue]):
    logger = logging.getLogger("frontend")
    logger.setLevel(logging.DEBUG)
    add_consolehandler(logger)
    add_filehandler(logger, "/home/xwh/new_vllm_docker/MatPlotAgent/log/frontend.log")
    with open('./benchmark_data/benchmark_instructions.json') as file:
        data = json.load(file)
    # loop = 300
    req = [50, 85, 34, 54, 53, 50, 49, 83, 42, 67, 30, 38, 70, 51, 98, 60, 69, 76, 71, 53, 58, 62, 48, 77, 13, 10, 44, 18, 32, 4, 99, 45, 89, 39, 7, 2, 24, 41, 31, 35, 84, 61, 28, 11, 27, 46, 66, 40, 63, 12, 95, 16, 80, 96, 6, 75, 81, 8, 78, 100, 36, 23, 5, 19, 21, 93, 15, 43, 79, 26, 86, 65, 56, 91, 14, 72, 73, 20, 33, 57, 9, 52, 68, 88, 25, 37, 17, 64, 87, 1, 59, 82, 29, 97, 92, 90, 74, 22, 47, 94]

    for ar_,cv_,slo_,pred in test: 
        np.random.seed(cv_seed[int(cv_)])
        gpu_model = list(GPU_MAP.keys())
        shape = 1 / cv_**2
        scale = cv_**2 / ar_
        interval = list(np.random.gamma(shape, scale, 300))# [:200]#[:100] * 3 #loop)
        logger.debug(f"Generated interval: {interval}") 
        warm_model = [0,1,2,3,4,5] # [:1]
        while warm_model:
            item = input_queue.get()
            if item not in warm_model:
                raise
            warm_model.remove(item)
        for model_id in gpu_model:
            output_queue[model_id].put((slo_,pred,))
        warm_model = gpu_model # [:1]
        assert warm_model[0] == 0
        while warm_model:
            item = input_queue.get()
            if item not in warm_model:
                raise
            warm_model.remove(item) 
          
        os.system("sh clean.sh")
        logger.info(f"Start test: arr:{ar_}, cv: {cv_}, slo: {slo_}, pred: {pred}")
        dat = []
        
        for i in range(loop):
            req_id = req[i % 100]
            exp_instruction = data[req_id - 1]["expert_instruction"]
            sim_instruction = data[req_id - 1]["simple_instruction"]

            directory = f'{workspace}/example_{i}' 
            if not os.path.exists(directory):
                os.mkdir(directory)
            else:
                raise
            if req_id in range(76,101):
                source_dir = f"./benchmark_data/data/{req_id}"
                dest_dir = f"{directory}"
                csv_files = glob.glob(f"{source_dir}/*.csv")
                for file in csv_files:
                    shutil.copy(file,dest_dir)
            input = {
                "req_id": i, #req_id,
                "query": exp_instruction,
                "budget": 0,
                "original": sim_instruction
            }
            dat.append(input)

        logger.info(f"All model warm up finished! Start serve: {time.time()}")
        for i in range(loop):
            time.sleep(interval[i])
            st = time.time()
            req_id = req[i % 100]
            data_dict = dat[i]
            data_dict["arrival_time"] = st
            data_dict["org_arrival_time"] = st
            data_dict["real_time"] = 0
            output_queue[0].put(data_dict)
            logger.debug(f"Prepare {req_id} take: {time.time()-st}")
        output_queue[0].put(None)

def get_model(input_queue: mp.Queue,output_queue: mp.Queue,model_id,config):
    model_class = config["model_config"][model_id]["class_name"]
    other_params = {}
    if "code" in model_class:
        return CodeExecutor(input_queue, output_queue,name=config["model_config"][model_id]["name"]), other_params
    elif "llm" in model_class:
        name = config["model_config"][model_id]["name"]
        if "codellama" in name:
            model_name = "/data/xwh/CodeLlama-7b-Instruct-hf"
            llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.90,
                disable_log_stats=True,
                max_model_len=4096,
                name=name,
                enable_chunked_prefill=args.chunk,
            )
            params = SamplingParams(temperature=0, top_p=1, max_tokens=1000,)
            other_params["pslo"] = 2
        else:
            model_name = "/data/xwh/llava-1.5-7b-hf"
            llm = LLM(
                model = model_name,
                tensor_parallel_size = 1,
                name=name,
                disable_log_stats=True,
            )
            params = SamplingParams(temperature=0, top_p=1, max_tokens=1000,)
            other_params["pslo"] = 2
        llm.llm_engine.scheduler[0].do_predict = config["do_predict"]
        other_params["sampling_params"] = params
        other_params["dslo"] = 2
        return llm, other_params
    else:
        raise NotImplementedError

def model_running(input_queue: mp.Queue, output_queue: mp.Queue, model_id: int, config, frontend_queue: mp.Queue=None):
    testt = [ t[0] for t in test]
    time_re = [ loop / t for t in testt]
    assert len(testt) == test_num, "Test re error!" 
    cur_gpu = GPU_MAP.get(model_id, None)
    if cur_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cur_gpu
    llm, other_params = get_model(input_queue, output_queue, model_id, config)
    logger = logging.getLogger(llm.name)
    logger.setLevel(logging.DEBUG)
    add_consolehandler(logger)
    add_filehandler(logger, llm.log_file)
    if model_id in GPU_MAP.keys():
        logger.info(f"{model_id} Start warm up! time: {time.time()}")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1, ignore_eos=True)
        for i in range(1,31):
            prompts_id = [1] +[ random.randint(10,100) for _ in range(32*i)]
            if i <= 10:
                for _ in range(50):
                    outputs = llm.generate(prompt_token_ids=[prompts_id]*2, sampling_params=sampling_params, use_tqdm=False)
            else:
                for _ in range(10):
                    outputs = llm.generate(prompt_token_ids=[prompts_id]*2, sampling_params=sampling_params, use_tqdm=False)
        logging.info("Start decode warm up!")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=500, ignore_eos=True)
        if model_id == 2 or model_id == 3:
            image = Image.open('test.jpg')
            inputs = {"prompt": "<image> What the image is?", 
                      "multi_modal_data": {
                        "image": image
                    }}
            outputs = llm.generate(inputs, sampling_params=sampling_params,use_tqdm=False)
        else:
            outputs = llm.generate(prompt_token_ids=[prompts_id]*2, sampling_params=sampling_params, use_tqdm=False)
        logger.info(f"{model_id} Warm up Finished! time: {time.time()}")

    for index in range(0,test_num):
        frontend_queue.put(model_id)
        if model_id in GPU_MAP.keys():
            item = input_queue.get()
            logger.info(f"item: {item}")
            cslo, pred = item
            other_params["dslo"] = cslo
            other_params["pslo"] = cslo
            llm.llm_engine.scheduler[0].do_predict = pred
            frontend_queue.put(model_id)

        outputs = llm.serve(input_queue, output_queue, **other_params)
        if llm.class_name == "CodeExecutor":
            real_time = deepcopy(outputs[1])
            req_id_list = deepcopy(outputs[2])
            req_re_list =deepcopy(outputs[3])
            outputs = outputs[0]
            slo = llm.slo
            array = np.array(outputs)
            satisfy = np.sum(array <= slo)
            summ = list(zip(req_id_list, real_time, outputs))
            logger.info(f"Finish test: {index}")
            logger.info(f"Code executor: sum: {summ} Total requests: {array.shape[0]}, satisfy: {satisfy}, ratio: {0 if array.shape[0] == 0 else satisfy/array.shape[0]:.4f}")
            logger.info(f"re-req: {req_re_list}")
        elif llm.class_name == "LLM":
            vios = 0
            attns = 0
            len_summary = []
            gprefill = 0
            gdecode = 0
            output_sum = []
            if outputs:
                start_time = min((output.arrival_time for output in outputs)) + 30 
                end_time = start_time + time_re[index] # -20 
                end_time_prefill = end_time

            for output in outputs:
                attained, info, vio, attn, _, e2e = llm.llm_engine.scheduler[0].predictor.check_seq_step_slo(output,end_time=end_time,start_time=start_time)
                vios += vio
                attns += attn
                gprefill += e2e[0]
                gdecode += e2e[1]
                
                len_summary.append((output.arrival_time-start_time, output.finished_time-start_time, output.req_id, len(output.prompt_token_ids), len(output.outputs[0].token_ids), output.finished_time - output.arrival_time, attn, vio))
            if outputs:
                gpt = attns / (end_time - start_time)
                gprefill_rate = gprefill / (end_time_prefill-start_time)
                gdecode_rate = gdecode / (end_time - start_time)
                logger.info(f"Finish test: {index}")
                logger.info(f"{llm.name}: Start time: {start_time} end time: {end_time} Length Summary: {len_summary}, good token num: {attns} , bad token num: {vios}, rate: {attns/(attns+vios) if attns + vios != 0 else 0}  , Goodput: {gpt:.2f} tokens/s, gprefill:{gprefill}, gdecode:{gdecode}, Goodprefill: {gprefill_rate:.2f} req/s, Gooddecode: {gdecode_rate:.2f} req/s")
        else:
            raise ValueError("Invalid Model")


if __name__ == '__main__':
    config = {
        "model_config": [
            {
                "class_name": "llm",
                "name": "codellama_1"
            },
            {
                "class_name": "code",
                "name": "code_executor_1"
            },
            {
                "class_name": "llm",
                "name": "llava_1"
            },
            {
                "class_name": "llm",
                "name": "llava_2"
            },
            {
                "class_name": "llm",
                "name": "codellama_2"
            },
            {
                "class_name": "code",
                "name": "code_executor_2"
            }
        ],
        "do_predict": True,
    }

    frontend_queue = mp.Queue()
    codellama_1 = mp.Queue() 
    code_executor_1 = mp.Queue() 
    llava_1 = mp.Queue()
    llava_2 = mp.Queue()
    codellama_2 = mp.Queue()
    code_executor_2 = mp.Queue()
    final_ouput = mp.Queue()

    process_ls = []

    p_frontend = mp.Process(target=frontend, args=(frontend_queue,[codellama_1,code_executor_1,llava_1,llava_2,codellama_2,code_executor_2]))
    # p_codellama_1 = mp.Process(target=model_running, args=(codellama_1,final_ouput, 0,config,frontend_queue))
    p_codellama_1 = mp.Process(target=model_running, args=(codellama_1,code_executor_1, 0,config,frontend_queue))

    p_code_executor_1 = mp.Process(target=model_running, args=(code_executor_1, [llava_1,llava_2], 1,config, frontend_queue))
    p_llava_1 = mp.Process(target=model_running,args=(llava_1, codellama_2, 2,config,frontend_queue))
    p_llava_2 = mp.Process(target=model_running,args=(llava_2, codellama_2, 3,config,frontend_queue))
    p_codellama_2 = mp.Process(target=model_running, args=(codellama_2,code_executor_2, 4,config,frontend_queue))
    p_code_executor_2 = mp.Process(target=model_running, args=(code_executor_2, final_ouput, 5,config, frontend_queue))

    process_ls.extend([p_frontend, p_codellama_1,p_code_executor_1, p_llava_1,p_llava_2, p_codellama_2, p_code_executor_2]) 
    for p in process_ls:
        p.start()
    for p in process_ls:
        p.join()
