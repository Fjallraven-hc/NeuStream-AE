import json
from typing import List
from vllm.core import predictor
import json
import numpy as np

class RequestResult:
    """
    A class for storing the results of a single request
    """
    
    def __init__(
        self,
        prompt_len: int,
        output_len: int,
        start_time: float,
        end_time: float,
        token_timestamps: List[float] = None,
        lifetime_events = None,
        output_text: str = None,
        step_time: List[float] = None,# step time是neu输出log里面，每个token结束的时间（包括prefill），token_timestamps一般是空
        timestamps_start: List[float] = None,
    ):
        
        self.drop = True if not token_timestamps else False
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.start_time = start_time
        self.end_time = end_time
        self.token_timestamps = token_timestamps
        self.lifecycle_events = lifetime_events
        self.step_time = step_time
        self.timestamps_start = timestamps_start
        self.ftl = self.step_time[0] - self.start_time

    def __repr__(self) -> str:
        return (f"RequestResult("
                f"prompt_len={self.prompt_len}, "
                f"output_len={self.output_len}, "
                f"start_time={self.start_time}, "
                f"end_time={self.end_time}, "
                f"ftl={self.ftl}"
                # f"step_time={self.step_time}, "
                # f"timestamps_start={self.timestamps_start})"
            )

def read_request_result(path: str) -> List[RequestResult]:
    with open(path, "r") as f:
        request_results: List[RequestResult] = [
            RequestResult(
                item["input_len"],
                item["output_len"],
                item["arrival_time"],
                item["step_time"][-1],
                step_time=item["step_time"],
            )
            for item in json.load(f) if item["output_len"] > 0
        ]
    return request_results

class CheckOutput:
    def __init__(
        self,
        overall_attained: bool,
        info: str,
        num_vio_tokens: int,
        num_attn_tokens: int,
        p_attained: bool,
        d_attained: bool,
    ) -> None:
        self.overall_attained = overall_attained
        self.info = info
        self.num_vio_tokens = num_vio_tokens
        self.num_attn_tokens = num_attn_tokens
        self.p_attained = p_attained
        self.d_attained = d_attained

class DistPredictor(predictor.Predictor):
    def check_seq_step_slo(self, output: RequestResult, pslo: float, dslo: float, p_strict_slo: bool = False):
        '''
            检查seq中每一步的SLO是否满足
        '''
        pmplen = output.prompt_len
        genlen = output.output_len
        num_vio = 0
        num_attained = 0
        prefill_attained = True
        e2e_attained = True

        ptime = output.ftl
        except_ptime = pslo*self.prefill_base_runtime(pmplen)
        if ptime > except_ptime:
            num_vio += 1
            prefill_attained = False
        else:
            num_attained += 1

        # strict prefill SLO
        if p_strict_slo and not prefill_attained:
            return CheckOutput(
                overall_attained=False,
                info="NULL MSG",
                num_vio_tokens=output.output_len,
                num_attn_tokens=num_attained,
                p_attained=prefill_attained,
                d_attained=False
            )
        
        for step in range(1, genlen):
            dtime = output.step_time[step] - output.start_time
            except_dtime = dslo*self.decode_stage_time_base(pmplen, step+1) + except_ptime
            if dtime > except_dtime:
                num_vio += 1
                if step == genlen - 1:
                    e2e_attained = False
                # return False, f"The {step} th decode stage timeout.(when generating the {step+1} token.) Expected use {except_dtime}, Actually {dtime}."
            else:
                num_attained += 1
        # return num_vio == 0, "NULL MSG", num_vio, num_attained, prefill_attained, e2e_attained
        return CheckOutput(
            overall_attained=(num_vio == 0),
            info="NULL MSG",
            num_vio_tokens=num_vio,
            num_attn_tokens=num_attained,
            p_attained=prefill_attained,
            d_attained=e2e_attained
        )


def str_ratio(n1, n2):
    if n2 == 0:
        return f"{n1}/{n2} = {0.0:.2f} %"
    return f"{n1}/{n2} = {n1*100/n2:.2f} %"

def get_neu_goodput(pred: DistPredictor, results: List[RequestResult], pslo: float, dslo: float, p_strict_slo: bool = False):
    check_outputs = [pred.check_seq_step_slo(result, pslo, dslo, p_strict_slo) for result in results]
    num_vio_tokens = sum([item.num_vio_tokens for item in check_outputs])
    num_attn_tokens = sum([item.num_attn_tokens for item in check_outputs])
    num_reqs = len(results)

    end_time = max([result.end_time for result in results])
    start_time = min([result.start_time for result in results])
    total_time = end_time - start_time
    goodput = num_attn_tokens / total_time
    throughput = (num_attn_tokens + num_vio_tokens) / total_time

    num_e2e_attained = sum([1 for item in check_outputs if item.d_attained])
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} token/s\n"
          f"e2e: {str_ratio(num_e2e_attained, num_reqs)}\n"
          f"pslo:{pslo}, dslo:{dslo}\n"
          f"Goodput: {goodput:.2f} token/s\n"
          f"ratio: {goodput/throughput*100:.2f}%")
    
    return check_outputs, goodput
    
def get_distllm_throughput(request_results: List[RequestResult]):
    end_time = max([item.end_time for item in request_results])
    start_time = min([item.start_time for item in request_results])
    benchmark_time = end_time - start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput:")
    print(f"Total tokens: {sum([req.output_len for req in request_results])}")
    print(f"\t{len(request_results) / benchmark_time:.2f} requests/s")
    print(f"\t{sum([req.prompt_len + req.output_len for req in request_results]) / benchmark_time:.2f} tokens/s")
    print(f"\t{sum([req.output_len for req in request_results]) / benchmark_time:.2f} output tokens/s")