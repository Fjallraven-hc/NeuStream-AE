from vllm import LLM, SamplingParams
from vllm.core.predictor import get_poisson_time, GammaProcess
from vllm.logger import init_logger, add_consolehandler, add_filehandler, output_slo_log, slo_chart
from vllm.core.predictor import LlamaPredictor
import numpy as np
import multiprocessing as mp
from typing import Optional
import time
import logging
import argparse
import json
import itertools

def str_ratio(n1, n2):
    if n2 == 0:
        return f"{n1}/{n2} = {0.0:.2f} %"
    return f"{n1}/{n2} = {n1*100/n2:.2f} %"

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0
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
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--dslo",
        type=float,
        default=1.5,
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


    return parser.parse_args()

def frontend(
        queue: mp.Queue,
        queue_llm: mp.Queue,
        rate: float, 
        total_time: float,
        seed: Optional[int],
        args,
    ):
    with open("../log/inputdataset3.txt", 'r') as f:
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
    seedmap = {1: 4, 1.5: 0, 2: 3, 2.5: 1, 3: 6, 3.5: 5, 4: 1, 4.5: 0}

    # 去除异常请求，这个请求仅仅生成10个token
    inputs[12] = inputs[11]
    inputs = inputs * 3

    logger = logging.getLogger("FRONTEND")
    logger.setLevel(logging.DEBUG)
    add_consolehandler(logger)
    add_filehandler(logger, f"./log/frontend.log")
    logger.info("front end get llm signal.")

    if args.gamma:
        logger.info("Use gamma process.")
    else:
        logger.info("Use possion process.")

    

    itertimes = queue_llm.get()
    for _ in range(itertimes):
        req_rate = queue_llm.get()
        cv = queue_llm.get()
        logger.info(f"req rate = {req_rate}")
        if args.gamma:
            logger.info(f"req cv = {cv}, seed = {seedmap[cv]}")
            gamma_generator = GammaProcess(req_rate, cv)
            timedeltas = gamma_generator.get_gamma_time(len(inputs), seedmap[cv])
        else:
            timedeltas = get_poisson_time(req_rate, rtmap[req_rate], seedmap[cv])
        
        arrival_time = np.cumsum(timedeltas)

        now = time.time()
        logger.info(f"start time: {now}")
        for delta, art, pmp in zip(timedeltas, arrival_time, inputs):
            time.sleep(delta)
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
    add_filehandler(logger, f"./log/llm-main.log")
    add_filehandler(logger, f"/home/v-yuanqwang/vllava/LLaVA/log/all.log")
    logger.debug("---"*30)
    logger.debug(f"rate = {rate} req/s, time = {times}s.")

    logger2 = logging.getLogger("step pred ")
    logger2.setLevel(logging.DEBUG)
    # add_consolehandler(logger2)
    add_filehandler(logger2, f"./log/step prediction.log")

    max_logger = logging.getLogger("max logger")
    max_logger.setLevel(logging.DEBUG)
    add_filehandler(max_logger, f"./log/step prediction max.log")
    

    sampling_params = SamplingParams(temperature=0,
                                    top_p=1,
                                    max_tokens=300)
    model_name = "facebook/opt-13b"
    llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.95,max_num_batched_tokens=10000, disable_log_stats=False)
    llm.llm_engine.scheduler.do_predict = not vllm_only
    predictor = llm.llm_engine.scheduler.predictor
    
    
    with open("./log/longprompt.txt") as f:
        longpmp = f.readline()
    llm.generate([longpmp, longpmp], sampling_params= SamplingParams(temperature=0, top_p=1, max_tokens=40))

    # rates = [1, 1.5, 1.6, 1.9]
    rates = [3, 4, 5]
    # rates.reverse()
    dslos = [1.5]
    pslos = [1.5]
    pred_steps = list(range(30, 30+1, 20))
    pred_step_times = [0.123]

    cvs = [1, 1.5, 2, 2.5]
    cvs = [1]
    if not args.gamma:
        cvs = [1]
    NUM_DUP = 5

    # 放入循环的次数
    queue_llm.put(len(rates)*len(pslos)*len(dslos)*len(pred_step_times)*len(pred_steps)*len(cvs)*NUM_DUP)

    for rate, pred_step_time, pslo, dslo, pred_step, cv in itertools.product(rates, pred_step_times, pslos, dslos, pred_steps, cvs):
        max_gpt = 0.0
        for _ in range(NUM_DUP):
            info = f"Schedule_enabled: {not args.vllm}, pslo: {pslo}, dslo: {dslo}, pred_step: {pred_step}, pred_step_time = {pred_step_time}, rate = {rate}, cv = {cv}"
            llm.llm_engine.log(info)
            logger2.debug(info)
            predictor.pred_step_time = pred_step_time
            predictor.pred_step = pred_step

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
            for output in outputs:
                step_times = [tm - output.arrival_time for tm in output.step_time]
                attained, info, vio, attn = predictor.check_seq_step_slo(output)
                vios += vio
                attns += attn
                attained_ls.append(attained)
                logger.info(f"Request {output.request_id:>4}: Overall Attained: {attained:>5}. Ratio: {str_ratio(attn, attn+vio)}")
                logger.debug(info)

            start_time = min((output.arrival_time for output in outputs))
            end_time = max((output.finished_time for output in outputs))

            logger.info(f"Actually SLO Attained ratio: {str_ratio(sum(attained_ls), num_req)}")
            logger.info(f"Tokenwise SLO attn: {str_ratio(attns, attns+vios)}")
            logger.info(f"Goodput: {attns/(end_time - start_time):.2f} tokens/s")
            logger.info(f"Total Time: {end_time - start_time:.2f} s")

            logger2.info(f"Actually SLO Attained ratio: {str_ratio(sum(attained_ls), num_req)}")
            logger2.info(f"Tokenwise SLO attn: {str_ratio(attns, attns+vios)}")
            gpt = attns/(end_time - start_time)
            max_gpt = max(max_gpt, gpt)
            logger2.info(f"Goodput: {gpt:.2f} tokens/s")
            logger2.info(f"Total Time: {end_time - start_time:.2f} s")

        max_logger.debug(f"pslo: {pslo}, dslo: {dslo}, pred_step: {pred_step}, pred_step_time = {pred_step_time}, rate = {rate}, cv = {cv}")
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
    

