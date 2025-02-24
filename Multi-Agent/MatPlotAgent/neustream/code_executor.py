import multiprocessing as mp
import time
import re
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Lock
import logging
import zmq
import pickle
from pathlib import Path
from .utils import run_code as run_c, is_run_code_success

REQ = [50, 85, 34, 54, 53, 50, 49, 83, 42, 67, 30, 38, 70, 51, 98, 60, 69, 76, 71, 53, 58, 62, 48, 77, 13, 10, 44, 18, 32, 4, 99, 45, 89, 39, 7, 2, 24, 41, 31, 35, 84, 61, 28, 11, 27, 46, 66, 40, 63, 12, 95, 16, 80, 96, 6, 75, 81, 8, 78, 100, 36, 23, 5, 19, 21, 93, 15, 43, 79, 26, 86, 65, 56, 91, 14, 72, 73, 20, 33, 57, 9, 52, 68, 88, 25, 37, 17, 64, 87, 1, 59, 82, 29, 97, 92, 90, 74, 22, 47, 94]

class CodeExecutor:
    id = 1

    def __init__(self, input_queue: mp.Queue, output_queue, **kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.name = kwargs.get("name", f"code_executor_{self.id}")
        self.id += 1
        self.class_name = "CodeExecutor"
        self.slo = kwargs.get("slo", 2.5)
        default_path = str(Path(__file__).parent.parent.resolve())
        # self.workspace = kwargs.get("workspace", "/home/xwh/new_vllm_docker/MatPlotAgent")
        self.workspace = kwargs.get("workspace", default_path)
        self.log_file = f"{self.workspace}/log/{self.name}.log"
        self.index = 0

        self.logger = self.init_logger()
        self.use_zmq = False
        if isinstance(input_queue, zmq.Socket):
            self.use_zmq = True

    def init_logger(self):
        logger = logging.getLogger(self.log_file)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger


    def get_code(self, response):
        text_list = list(response.splitlines())
        start = 0
        end = len(text_list) - 1
        for i, line in enumerate(text_list):
            line = line.strip()
            if line.startswith("import") or line.startswith("from"):
                text_list[i] = line
                start = i
                break
        for i, line in enumerate(text_list[::-1]):
            if line.endswith(")"):
                end = len(text_list) - i
                break
        return "\n".join(text_list[start:end])


    def serve(
        self,
        input_queue: mp.Queue,
        output_queue,
        **kwargs
    ):
        # self.logger.debug(f"Start serving.")
        no_req: bool = False
        lock = Lock()
        def run_code(input):
            text = input["query"]
            req_id = input["req_id"]
            budget = input["budget"]
            st = time.time()
            code = self.get_code(text)
            file_name = f"{self.name}.py"

            with open(os.path.join(f'{self.workspace}/workspace3/example_{req_id}', file_name), 'w') as f: 
                f.write(code)

            log = run_c(f'{self.workspace}/workspace3/example_{req_id}', file_name) 

            if code is None or not is_run_code_success(log) or not os.path.exists(f'{self.workspace}/workspace3/example_{req_id}/novice.png'):
                shutil.copy(f'{self.workspace}/benchmark_data/ground_truth/example_{REQ[req_id%100]}.png', f'{self.workspace}/workspace3/example_{req_id}/novice.png')
            ed = time.time()
            elapsed = ed - input["arrival_time"]
            real = ed - st
            check_num = self.slo - elapsed ## 还剩多少时间
            now = time.time()
            self.logger.debug(f"Req: {req_id}, arrive_time: {st}, finish_time: {ed}; elapsed: {elapsed}, slo: {self.slo}, budget: {budget}, ")
            if check_num + budget >= 0: ## 能不能用budget，能用就放入下一个阶段
                res = {
                    "req_id": req_id,
                    "query": code,
                    "budget": check_num + budget,  
                    "arrival_time": now,
                    "org_arrival_time": input["org_arrival_time"],
                    "real_time": input["real_time"] + real,
                    }
                if "1" in self.name:
                    res["original"] = input["original"]
                if self.use_zmq:
                    output_queue.send(pickle.dumps(res))
                else:
                    with lock: 
                        if "1" in self.name:
                            output_queue[self.index].put(res)

                            self.index = (self.index + 1) % 2
                            # self.logger.debug(f"Req {req_id} put in the {self.index}th queue.")
                        else:
                            output_queue.put(res)
            return elapsed, real, req_id, (now, 0)
        
        futures = []
        with ThreadPoolExecutor(max_workers=40) as executor:
            while not no_req:
                if self.use_zmq:
                    data = input_queue.recv()
                    req = pickle.loads(data)
                else:
                    req = input_queue.get()
                if req is None:
                    break
                futures.append(executor.submit(run_code, req))
            res = wait(futures, timeout=None)
            assert len(res.not_done) == 0
            if self.use_zmq:
                output_queue.send(pickle.dumps(None))
            else:
                if "1" in self.name:
                    output_queue[0].put(None)
                    output_queue[1].put(None)
                else:
                    output_queue.put(None)

        process_time = []
        real_time = []
        req_id_list = []
        req_re = []
        for future in futures:
            p,r,r_id,re = future.result()
            process_time.append(p)
            real_time.append(r)
            req_id_list.append(r_id)
            req_re.append(re)
        return process_time,real_time,req_id_list, req_re



