import multiprocessing as mp
import time
import re
import os
import shutil
from ..utils import run_code, is_run_code_success

class CodeExecutor:
    id = 1

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, **kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.name = kwargs.get("name", f"code_executor_{self.id}")
        self.id += 1
        self.class_name = "CodeExecutor"
        self.slo = kwargs.get("slo", 1.5)
        self.workspace = kwargs.get("workspace", "/home/xwh/MatPlotAgent")

    def get_code(self, response):
        all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)
        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        return all_code_blocks_combined

    def run_code(self,input):
        text = input["prompt"]
        req_id = input["req_id"]
        code = self.get_code(text)

        os.makedirs(f'{self.workspace}/workspace2/{req_id}', exist_ok=True)

        file_name = f'workspace2/{req_id}/code_execution_{self.id}.py'
        with open(os.path.join(self.workspace, file_name), 'w') as f:
            f.write(code)
        log = run_code(self.workspace, file_name)     
        return log, code

    def serve(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        **kwargs
    ):
        no_req: bool = False
        process_time = []
        while not no_req:
            req = input_queue.get()
            if req is None:
                break
            req_id = req["req_id"]
            log, code = self.run_code(req)
            if not is_run_code_success(log):
                shutil.copy(f'{self.workspace}/benchmark_data/ground_truth/example_{req_id}.png', f'{self.workspace}/workspace2/{req_id}/novice.png')
            elapsed = time.time() - req["arrival_time"]
            process_time.append(elapsed)
            if elapsed <= self.slo:
                res = {
                    "req_id": req_id,
                    "prompt": "hello",
                    "budget": req["budget"],  
                }
                output_queue.put(res)
        return process_time









