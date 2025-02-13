'''
首先，/home/msra/wyq/vllm_files/vllm-0.5.4/vllm/distributed/device_communicators/custom_all_reduce_utils.py是改过的。
基于https://github.com/vllm-project/vllm/issues/7846
其次设置了cuda=11.8
最后longprompt2.txt也改动了

'''

from vllm import LLM, SamplingParams
from vllm.core.predictor import get_poisson_time, GammaProcess
from vllm.logger import init_logger, add_consolehandler, add_filehandler, output_slo_log, slo_chart, LOG_DIR
from vllm.outputs import RequestOutput

import numpy as np
import multiprocessing as mp
from typing import Optional
import time
import logging
import argparse
import json
import itertools
from utils_data import sample_requests, Dataset, TestRequest
from typing import List


def repr_output(output: RequestOutput):
    return {
        "request_id": output.request_id,
        "input_len": len(output.prompt_token_ids),
        "output_len": len(output.outputs[0].token_ids),
        "arrival_time": output.arrival_time,
        "step_time": output.step_time,
        "p_start_time": output.p_start_time,
        # "timestamps": output.timestamps,
    }



def str_ratio(n1, n2):
    if n2 == 0:
        return f"{n1}/{n2} = {0.0:.2f} %"
    return f"{n1}/{n2} = {n1*100/n2:.2f} %"

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--vllm",
        action='store_true'
    )

    parser.add_argument(
        "--pslo",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dslo",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pred_step",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--pred_step_time",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--gamma",
        action='store_true'
    )
    parser.add_argument(
        "--dataset", type=str, default="/home/msra/wyq/DistServe/dataset/sharegpt.ds", help="Path to the (preprocessed) dataset."
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--round_robin_id",
        "-rid",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--round_robin_worldsize",
        "-rws",
        type=int,
        default=1,
    )

    return parser.parse_args()

def frontend(
        queue: mp.Queue,
        queue_llm: mp.Queue,
        rate: float, 
        total_time: float,
        seed: Optional[int],
        args,
    ):
    with open("./inputs/inputdataset3.txt", 'r') as f:
        inputs = json.load(f)

    rtmap = {
        1 :100,
        1.5 :66,
        1.6 :70,
        1.7 :70,
        1.9 :70,
        2 :50,
        3 :33,
        4 :25,
        5 :20,
    }

    # cv->np.seed
    # 设置它的原因是，某些seed在100这个尺度下的总时间跟相关的速率相差甚远。这些seed生成的速率在总数100个的情况下，平均间隔与速率要求的比较接近
    seedmap = {1: 4, 
               1.5: 0, 
               2: 3, 
               2.5: 1, 
               3: 6, 
               3.5: 5, 
               4: 1, 
               4.5: 0}

    # 去除异常请求，这个请求仅仅生成10个token
    inputs[12] = inputs[11]

    inputs = [
        TestRequest(prompt=pmp, prompt_len=0, output_len=args.max_tokens) for pmp in inputs
    ]
    inputs = inputs * 10



    # num_req = 500
    # input_requests = sample_requests(
    #     args.dataset, num_req, seed=0
    # )
    # inputs = input_requests


    print(f"len inputs: {len(inputs)}")

    logger = logging.getLogger("FRONTEND")
    logger.setLevel(logging.DEBUG)
    add_consolehandler(logger)
    add_filehandler(logger, f"frontend.log")
    logger.info("front end get llm signal.")

    if args.gamma:
        logger.info("Use gamma process.")
    else:
        logger.info("Use possion process.")

    

    cv_seed = 0
    itertimes = queue_llm.get()
    for _ in range(itertimes):
        req_rate = queue_llm.get()
        cv = queue_llm.get()
        logger.info(f"req rate = {req_rate}")
        if args.gamma:
            logger.info(f"req cv = {cv}, seed = {seedmap[cv]}")
            gamma_generator = GammaProcess(req_rate, cv)
            timedeltas = gamma_generator.get_gamma_time(len(inputs), seedmap[cv])
            # timedeltas = gamma_generator.get_gamma_time(len(inputs), cv_seed)
        else:
            timedeltas = get_poisson_time(req_rate, rtmap[req_rate], seedmap[cv])

        if args.round_robin_worldsize > 1:
            inputs = inputs[args.round_robin_id::args.round_robin_worldsize]
            timedelta_ = []
            start = 0
            end = args.round_robin_id + 1
            while True:
                if start >= len(timedeltas):
                    break
                timedelta_.append(timedeltas[start:end].sum())
                start = end
                end += args.round_robin_worldsize

            timedeltas = np.array(timedelta_)
            logger.info(f"round robin id: {args.round_robin_id}, worldsize: {args.round_robin_worldsize}, len(timedeltas): {len(timedeltas)}, len(inputs): {len(inputs)}")
            logger.info(f"timedeltas sum: {sum(timedeltas)}")
        
        arrival_time = np.cumsum(timedeltas)
        
        
        now = time.time()
        logger.info(f"start time: {now}")
        for delta, art, pmp in zip(timedeltas, arrival_time, inputs):
            time.sleep(delta)
            # change to put request to ip:port
            queue.put(pmp)
            # logger.info(f"real delta = {time.time() - now}, exception: {art}")
        queue.put(None)

def llm_running(
        queue: mp.Queue,
        queue_llm: mp.Queue,
        rate: int,
        times: int,
        vllm_only: bool,
        # pslo: float,
        # dslo: float,
        # pred_step: int,
        # pred_step_time: float,
        args,
):
    logger = logging.getLogger("LLM running")
    logger.setLevel(logging.DEBUG)
    add_consolehandler(logger)
    add_filehandler(logger, f"llm-main.log")
    add_filehandler(logger, f"all.log")
    logger.debug("---"*30)
    logger.debug(f"rate = {rate} req/s, time = {times}s.")

    logger2 = logging.getLogger("step pred ")
    logger2.setLevel(logging.DEBUG)
    # add_consolehandler(logger2)
    add_filehandler(logger2, f"step prediction.log")

    logger3 = logging.getLogger("queuetime")
    logger3.setLevel(logging.INFO)
    add_filehandler(logger3, f"queuetime.log")

    max_logger = logging.getLogger("max logger")
    max_logger.setLevel(logging.DEBUG)
    add_filehandler(max_logger, f"step prediction max.log")
    

    sampling_params = SamplingParams(temperature=0,
                                    top_p=1,
                                    max_tokens=args.max_tokens)
    # model_name = "facebook/opt-30b"
    # model_name = "facebook/opt-66b"
    model_name = "facebook/opt-13b"
    llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9,max_num_batched_tokens=8192, disable_log_stats=False)
    llm.llm_engine.scheduler[0].do_predict = not vllm_only
    predictor = llm.llm_engine.scheduler[0].predictor
    
    
    with open("./inputs/longprompt2.txt") as f:
        longpmp = f.readline()
    llm.generate([longpmp, longpmp], sampling_params= SamplingParams(temperature=0, top_p=1, max_tokens=40))

    # rates = [1, 1.5, 1.6, 1.9]
    rates = [float(rt) for rt in args.rate.split(',')] if args.rate else [4]
    rates.reverse()
    dslos = [float(dslo) for dslo in args.dslo.split(',')] if args.dslo else [1.5]
    pslos = [float(pslo) for pslo in args.pslo.split(',')] if args.pslo else [3]
    pred_steps = [30]
    pred_genlens = [None]
    pred_step_times = [0.123]

    cvs = [float(cv) for cv in args.cv.split(',')] if args.cv else [1]
    if not args.gamma:
        cvs = [1]
    NUM_DUP = 5

    # 放入循环的次数
    queue_llm.put(len(rates)*len(pslos)*len(dslos)*len(pred_step_times)*len(pred_steps)*len(pred_genlens)*len(cvs)*NUM_DUP)

    for pred_genlen, rate, pred_step_time, pslo, dslo, pred_step, cv in itertools.product(pred_genlens, rates, pred_step_times, pslos, dslos, pred_steps, cvs):
        max_gpt = 0.0
        for _ in range(NUM_DUP):
            info = f"model: {model_name}, Schedule_enabled: {not args.vllm}, pslo: {pslo}, dslo: {dslo}, pred_step: {pred_step}, pred_step_time = {pred_step_time}, pred_genlen = {pred_genlen}, rate = {rate}, cv = {cv}" + (f", round robin:{args.round_robin_id+1}/{args.round_robin_worldsize}" if args.round_robin_worldsize > 1 else "")
            llm.llm_engine.log(info)
            logger2.debug(info)
            predictor.pred_step_time = pred_step_time
            predictor.pred_step = pred_step
            predictor.pred_step_p = pred_step
            llm.llm_engine.scheduler[0].total_sche_time = 0.0
            llm.llm_engine.total_step_time = 0.0
            llm.llm_engine.total_prefill_time = 0.0

            # 本次推理开始，设定请求速度
            queue_llm.put(rate)
            queue_llm.put(cv)
            outputs = llm.serve(queue, pslo, dslo, sampling_params)
            tokenizer = llm.get_tokenizer()
            

            table, info, num_req = slo_chart(outputs, predictor, tokenizer, rate*times)
            logger.debug('\n'+table)
            logger.debug(info)

            attained_ls = []
            vios = 0
            attns = 0
            queue_time_ls = []
            p_time_ls = []
            num_p_attain, num_d_attain = 0, 0
            for output in outputs:
                step_times = [tm - output.arrival_time for tm in output.step_time]
                if output.p_start_time is not None:
                    queue_time_ls.append(output.p_start_time - output.arrival_time)
                    p_time_ls.append(output.step_time[0] - output.arrival_time)
                attained, info, vio, attn, p_attain, d_attain = predictor.check_seq_step_slo(output)
                vios += vio
                attns += attn
                if p_attain:
                    num_p_attain += 1
                if d_attain:
                    num_d_attain += 1
                attained_ls.append(attained)
                logger.info(f"Request {output.request_id:>4}: Overall Attained: {attained:>5}. p: {p_attain:>5}, d: {d_attain:>5} Ratio: {str_ratio(attn, attn+vio)}")
                # logger.debug(info)

            output_texts = [output.outputs[0].text for output in outputs]
            # if LOG_DIR:
            #     import os
            #     import json
            #     with open(os.path.join(LOG_DIR, "output.txt"), "w") as f:
            #         json.dump(output_texts, f)
                
            #     result_log_dir = os.path.join(LOG_DIR, "result_log")
            #     os.makedirs(result_log_dir, exist_ok=True)
            #     with open(os.path.join(result_log_dir, f"rate_{rate}-cv_{cv}-slo_{pslo}+{dslo}-predstep_{pred_step}.log"), "w") as f:
            #         json.dump(outputs, f, default=repr_output)

            start_time = min((output.arrival_time for output in outputs))
            end_time = max((output.finished_time for output in outputs))

            logger.info(f"wait time len: {len(queue_time_ls)}. max: {max(queue_time_ls)}, min: {min(queue_time_ls)}.")
            logger.info(f"Actually SLO Attained ratio: {str_ratio(sum(attained_ls), num_req)}")
            logger.info(f"Tokenwise SLO attn: {str_ratio(attns, attns+vios)}")
            logger.info(f"serve reqs: {len(queue_time_ls)}, p attain: {str_ratio(num_p_attain, len(queue_time_ls))}, d attain: {str_ratio(num_d_attain, len(queue_time_ls))}")
            logger.info(f"p goodput: {num_p_attain / (end_time - start_time) :.2f} req/s, d goodput: {num_d_attain / (end_time - start_time) :.2f} req/s,")
            logger.info(f"Goodput: {attns/(end_time - start_time):.2f} tokens/s")
            logger.info(f"Total Time: {end_time - start_time:.2f} s")
            logger.info(f"Sche Time: {llm.llm_engine.scheduler[0].total_sche_time:.2f} s")
            logger.info(f"Step Time: {llm.llm_engine.total_step_time:.2f} s")
            logger.info(f"Prefill Time: {llm.llm_engine.total_prefill_time:.2f} s, p:d ratio = 1:{llm.llm_engine.total_step_time/llm.llm_engine.total_prefill_time:.2f}")

            logger2.info(f"wait time len: {len(queue_time_ls)}. max: {max(queue_time_ls)}, min: {min(queue_time_ls)}.")
            logger2.info(f"Actually SLO Attained ratio: {str_ratio(sum(attained_ls), num_req)}")
            logger2.info(f"serve reqs: {len(queue_time_ls)}, p attain: {str_ratio(num_p_attain, len(queue_time_ls))}, d attain: {str_ratio(num_d_attain, len(queue_time_ls))}")
            logger2.info(f"p goodput: {num_p_attain / (end_time - start_time) :.2f} req/s, d goodput: {num_d_attain / (end_time - start_time) :.2f} req/s,")
            logger2.info(f"Tokenwise SLO attn: {str_ratio(attns, attns+vios)}")
            gpt = attns/(end_time - start_time)
            max_gpt = max(max_gpt, gpt)
            logger2.info(f"Goodput: {gpt:.2f} tokens/s")
            logger2.info(f"Total Time: {end_time - start_time:.2f} s")
            logger2.info(f"Sche Time: {llm.llm_engine.scheduler[0].total_sche_time:.2f} s")
            logger2.info(f"Step Time: {llm.llm_engine.total_step_time:.2f} s")
            logger2.info(f"Prefill Time: {llm.llm_engine.total_prefill_time:.2f} s, p:d ratio = 1:{llm.llm_engine.total_step_time/llm.llm_engine.total_prefill_time:.2f}")
            logger2.info(f"len(sche_time): {len(llm.llm_engine.scheduler[0].sche_time)}, len(decode_time): {len(llm.llm_engine.scheduler[0].sche_decode_time)}")
            mx = -100
            sche, dec = 0, 0
            for s, d in zip(llm.llm_engine.scheduler[0].sche_time, llm.llm_engine.scheduler[0].sche_decode_time):
                if s/d > mx:
                    mx = s/d
                    sche = s
                    dec = d
            logger2.info(f"max sche portion: {mx*100} %, s = {sche * 1000} ms, d = {dec * 1000} ms.")

            logger3.info(f"{queue_time_ls}\n\n\n")
            logger3.info(f"{p_time_ls}\n\n\n")
            logger3.info("-"*100)

        max_logger.debug(f"pslo: {pslo}, dslo: {dslo}, pred_step: {pred_step}, pred_step_time = {pred_step_time}, pred_genlen = {pred_genlen}, rate = {rate}, cv = {cv}"+ (f", round robin:{args.round_robin_id+1}/{args.round_robin_worldsize}" if args.round_robin_worldsize > 1 else ""))
        max_logger.info(f"Goodput: {max_gpt:.2f} tokens/s")

    



if __name__ == '__main__':
    print(__file__)
    args = _parse_args()
    queue = mp.Queue()
    queue_sig = mp.Queue()
    process_ls = []
    p_front = mp.Process(target=frontend, args=(queue, queue_sig, args.rate, args.time, args.seed, args))
    p_llm = mp.Process(target=llm_running, args=(queue, queue_sig, args.rate, args.time, args.vllm, args)) # args.pslo, args.dslo, args.pred_step, args.pred_step_time

    process_ls.extend([p_front, p_llm])

    for p in process_ls:
        p.start()
    for p in process_ls:
        p.join()
    

