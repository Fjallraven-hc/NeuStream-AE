from typing import List, Dict, Optional, Deque
import numpy as np


from vllm.logger import init_logger, add_consolehandler, add_filehandler
from vllm.sequence import SequenceGroup
from vllm.outputs import RequestOutput
import logging
import os
from pathlib import Path

logger = init_logger("Predictor")
logger.setLevel(logging.DEBUG)
add_consolehandler(logger, logging.INFO)
path = str(Path(__file__).parent.parent.parent.parent.parent.resolve())
add_filehandler(logger,os.path.abspath(f"{path}/MatPlotAgent/log/predictor.log"), logging.DEBUG)
add_filehandler(logger, os.path.abspath(f"{path}/MatPlotAgent/log/all.log"))
logger.info("Predictor Imported." + "-"*40)


class Predictor:
    length_predict = Dict[int, int]
    def _predict_len(self, prompt_len, generate_len) -> int:
        return self.length_predict[prompt_len+generate_len]
    
    def predict_runtime(seq_groups: List[SequenceGroup], stage: str):
        raise NotImplementedError
    
class LlamaPredictor(Predictor):
    def __init__(self, alpha, p_time, p_para, d_para) -> None:
        super().__init__()
        self.alpha = alpha
        self.length_predict = {i:1 if i < 33 else 1 for i in range(0, 10000)}
        self.lenmeta = 32
        self.prefill_time = np.array(p_time)

        self.timemodle_para_p = p_para
        self.timemodle_para_d = d_para
        self.pred_step_time = 0.01
        self.pred_step = 0
        self.pred_step_p = 10

        self.decision_histoty: List[int] = []
        self.step_time_history: List[float] = []

        self.last_decision: int = 0
        self.last_decode_time: float = 0.016 # 上一步decode用了多少时间？
        self.base_decode_time: float = 0.016
    
    def prefill_base_runtime(self, prompt_len: int) -> float:
        # 计算bs=1的时候，该长度的prompt的prefill阶段时间
        # print(f"prompt len: {prompt_len}")
        return self.prefill_time[(prompt_len+self.lenmeta-1)//self.lenmeta - 1]
    
    def decode_base_runtime(self, prompt_len: int, gen_len: int, pred_len: int):
        # 计算bs=1的时候， 初始长度prefix_len，已生成gen_len，预测再生成pred_len需要多少时间
        return self.timemodle_para_d[0] + self.timemodle_para_d[1]*(prompt_len+gen_len)
    
    def _decode_stage_time_base(self, prompt_len: int, gen_len: int):
        '''
        genlen=1, return 0
        因为prefill后genlen=1
        genlen=2为第一步decode完成后。
        '''
        res = 0
        for i in range(1, gen_len):
            res += self.timemodle_para_d[0] + self.timemodle_para_d[1]*(prompt_len+i)
        return res
    
    def decode_stage_time_base(self, prompt_len: int, gen_len: int):
        '''
        上一个函数去除循环的版本
        '''
        items = gen_len-1
        res = self.timemodle_para_d[0]*items + self.timemodle_para_d[1]*(items*(items+1)/2 + prompt_len*items)
        return res

    def predict_len(self, seq_group: SequenceGroup) -> int:
        seq = seq_group.get_unfinished_seqs()[0]
        return self._predict_len(seq.get_prompt_len(), seq.get_output_len())
    
    def predict_runtime(self, lengths: np.ndarray, stage: str):
        # TODO: runtime perf
        if stage == "prefill":
            return self.timemodle_para_p[0] + self.timemodle_para_p[1]*lengths.sum() + self.timemodle_para_p[2]*(lengths**2).sum()
        elif stage == "decode":
            # 经验误差
            res = self.timemodle_para_d[0] + self.timemodle_para_d[1]*lengths.sum()
            ress = res * 1e3
            if ress < 45:
                return res
            elif 45 < ress < 50:
                return res + 2e-3
            elif 50 < ress < 55:
                return res + 3e-3
            elif 55 < ress:
                return res + 4e-3
        else:
            raise NotImplementedError
    
    def predict_runtime_cumsum(self, lengths: np.ndarray, stage: str):
        if stage == "prefill":
            res = self.timemodle_para_p[0] + self.timemodle_para_p[1]*lengths.cumsum() + self.timemodle_para_p[2]*(lengths**2).cumsum()
            len_sum = lengths.cumsum()
            ids = (len_sum.astype(int) + 31) // 32 - 1
            mask = ids < 64
            res[mask] = self.prefill_time[ids[mask]]
            return res
        elif stage == "decode":
            return self.timemodle_para_d[0] + self.timemodle_para_d[1]*lengths.cumsum()
        else:
            raise NotImplementedError
        
    def get_last_budget(self, dslo:float):
        return dslo*self.base_decode_time - self.last_decode_time

    def check_batch_slo(
            self,
            time_rest: List[float],
            prompt_lengths: List[int],
            output_lengths: List[int],
            predict_lengths: List[int],
            stage: str,
        ) -> bool:
            # return True
            '''
            检查输入的batch信息能否满足所有的SLO
            '''
            if len(time_rest) == 0:
                return True
            time_rest = np.array(time_rest)
            prompt_lengths = np.array(prompt_lengths)
            output_lengths = np.array(output_lengths)
            lengths = prompt_lengths + output_lengths

            if stage == "prefill":
                time_rest -= self.predict_runtime(lengths=lengths, stage=stage)
                if any(time_rest < 0):
                    return False
            elif stage == "decode":
                predict_lengths = np.array(predict_lengths)
                running_mask = (predict_lengths > 0)
                max_pred_len = predict_lengths.max().item()
                logger.debug(f"max pred len = {max_pred_len}")
                for step in range(1, max_pred_len+1):
                    l1 = lengths[running_mask]
                    pr = self.predict_runtime(lengths=l1, stage=stage)
                    logger.debug(f"predict decode running time: {pr}")
                    time_rest -= pr
                    logger.debug(f"time_rest after: {time_rest}")
                    if any(time_rest[running_mask] < 0):
                        return False
                    running_mask = predict_lengths > step
                    lengths += 1
            else:
                raise NotImplementedError
            
            return True

    def get_max_batch(
            self,
            seq_groups: List[SequenceGroup],
            stage: str,
            now: float,
        ):
        if len(seq_groups) == 0:
            logger.info(f"No request in {stage} stage.")
            return []
        max_batch = []
        prompt_lengths = []
        output_lengths = []
        predict_lengths = []
        time_rest = []
        logger.info(f"get max batch of {stage} stage.")
        for seq_group in seq_groups:
            req = seq_group.get_unfinished_seqs()[0]
            prompt_length = req.get_prompt_len()
            output_length = req.get_output_len()
            predict_length = self.predict_len(seq_group=seq_group)

            prompt_lengths.append(prompt_length)
            output_lengths.append(output_length)
            predict_lengths.append(predict_length)

            if stage == "prefill":
                rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo - now
                time_rest.append(rest_time)
            elif stage == "decode":
                rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo + self.decode_base_runtime(prompt_length, output_length, predict_length) * seq_group.decode_slo - now
                time_rest.append(rest_time)
            else:
                raise NotImplementedError
            
            if not self.check_batch_slo(time_rest=time_rest, prompt_lengths=prompt_lengths, output_lengths=output_lengths, predict_lengths=predict_lengths, stage=stage):
                break
            max_batch.append(seq_group)

        return max_batch

    def get_max_run_batch_biserch(
            self,
            seq_groups: List[SequenceGroup],
            stage: str,
            now: float,
    ):
        '''
        Input: 
        seq_groups:待调度的seq_group列表
        stage:阶段
        now:当前时间
        Return:
        该函数首先检查是否
        '''
        timeout_seq_groups: List[SequenceGroup] = []
        ontime_seq_groups: List[SequenceGroup] = []

        len_input = len(seq_groups)

        prompt_lengths = []
        output_lengths = []
        predict_lengths = []
        time_rest = []

        while seq_groups:
            seq_group = seq_groups[0]
            
            req = seq_group.get_unfinished_seqs()[0]
            prompt_length = req.get_prompt_len()
            
            output_length = req.get_output_len()
            predict_length = self.predict_len(seq_group=seq_group)

            if stage == 'prefill':
                rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo - now + max(0, (30 - req.get_output_len())) * 0.022
            elif stage == 'decode':
                rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo + self.decode_base_runtime(prompt_length, output_length, predict_length) * seq_group.decode_slo - now + max(0, (30 - req.get_output_len())) * 0.022
            else:
                raise NotImplementedError
            
            if rest_time < self.prefill_base_runtime(prompt_length):
                logger.debug(f'Drop Request {seq_group.request_id} because it is timeout.')
                timeout_seq_groups.append(seq_group)
                seq_groups.pop(0)
            else:
                prompt_lengths.append(prompt_length)
                output_lengths.append(output_length)
                predict_lengths.append(predict_length)
                time_rest.append(rest_time)
                ontime_seq_groups.append(seq_group)
                seq_groups.pop(0)

        
        assert(len(ontime_seq_groups) + len(timeout_seq_groups) == len_input)
                

        bs, l, r = 0, 0, len(ontime_seq_groups)
        while l <= r:
            m = (l+r) // 2
            if self.check_batch_slo(time_rest=time_rest[:m], prompt_lengths=prompt_lengths[:m], output_lengths=output_lengths[:m], predict_lengths=predict_lengths[:m], stage=stage):
                bs = max(bs, m)
                l = m + 1
            else:
                r = m - 1
        return m, ontime_seq_groups, timeout_seq_groups
    
    def get_punctual_requests(
            self,
            seq_groups: List[SequenceGroup],
            now: float,
            stage: str = 'prefill',
    ):
        
        assert stage == 'prefill'
        timeout_seq_groups: List[SequenceGroup] = []
        punctual_seq_groups: List[SequenceGroup] = []
        while seq_groups:
            seq_group = seq_groups[0]
            req = seq_group.get_unfinished_seqs()[0]
            prompt_length = req.get_prompt_len()
            ## FIX 注意这里应该考虑请求的budget
            rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo - now  + max(0, (self.pred_step_p - req.get_output_len())) * self.get_last_budget(seq_group.decode_slo)

            if rest_time < self.prefill_base_runtime(prompt_length):
                logger.debug(f'Drop Request {seq_group.request_id} because it is timeout. rest time: {rest_time*1000:.1f} ms, base run time:{self.prefill_base_runtime(prompt_length)*1000:.1f} ms.')
                timeout_seq_groups.append(seq_group) 
                seq_groups.pop(0)
            else:
                punctual_seq_groups.append(seq_group)
                seq_groups.pop(0)
        
        return punctual_seq_groups, timeout_seq_groups

    def switch_P(
            self,
            prefill_reqs: Deque[SequenceGroup],
            decode_reqs: Deque[SequenceGroup],
            now: float,
    ):
        ontime_seq_groups, timeout_seq_groups = self.get_punctual_requests(list(prefill_reqs), now)

        # max_prefill_batch = ontime_seq_groups
        
        pmp_lens = np.array([seq_group.get_unfinished_seqs()[0].get_prompt_len() for seq_group in ontime_seq_groups])
        # pmp_timecost = self.predict_runtime(pmp_lens, "prefill")
        pmp_timecosts = self.predict_runtime_cumsum(pmp_lens, "prefill")

        prompt_lengths = []
        output_lengths = []
        predict_lengths = []
        time_rest = []

        for seq_group in decode_reqs:
            req = seq_group.get_unfinished_seqs()[0]
            prompt_length = req.get_prompt_len()
            output_length = req.get_output_len()
            predict_length = self.predict_len(seq_group=seq_group)

            prompt_lengths.append(prompt_length)
            output_lengths.append(output_length)
            predict_lengths.append(predict_length)

            ## FIX
            rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo + self.decode_stage_time_base(prompt_length, output_length+1) * seq_group.decode_slo - now + max(0, (self.pred_step - req.get_output_len())) * self.get_last_budget(seq_group.decode_slo)
            # 22ms是SLO为1.5的时候，每步会产生的富余。这里先给每个生成长度小于30的计算富余
            # 注意，这里没有减去pmp_timecost，这是因为pmp的时间需要再下面二分搜索的时候才能确定，下面也是
            time_rest.append(rest_time)

        logger.debug(f"Decode length sum: {sum(prompt_lengths) + sum(output_lengths)}")
        if len(decode_reqs[0].step_time) > 1:
            logger.debug(f"last step time: {(decode_reqs[0].step_time[-1] - decode_reqs[0].step_time[-2])*1000:.1f} ms")
        logger.debug(f"predict step time: {self.predict_runtime(np.array(prompt_lengths) + np.array(output_lengths), 'decode') * 1000:.1f} ms")
        logger.debug(f"Original Decode Time Rest: {time_rest}")

        len_decode = len(decode_reqs)

        # test:短请求优先是否能获得更好的表现
        ontime_seq_groups = sorted(ontime_seq_groups, key=lambda x: x.get_unfinished_seqs()[0].get_prompt_len())
        
        for seq_group in ontime_seq_groups:
            req = seq_group.get_unfinished_seqs()[0]
            prompt_length = req.get_prompt_len()
            output_length = 1
            predict_length = self.predict_len(seq_group=seq_group)

            prompt_lengths.append(prompt_length)
            output_lengths.append(output_length)
            predict_lengths.append(predict_length)

            # 此处没有减去pmp_timecost
            rest_time = seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo + self.decode_stage_time_base(prompt_length, output_length+1) * seq_group.decode_slo - now + max(0, (self.pred_step_p - req.get_output_len())) * self.get_last_budget(seq_group.decode_slo)
            time_rest.append(rest_time)

            logger.debug(f"Seq {seq_group.request_id:>4}: rest time:{(seq_group.budget + seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo - now)*1000:.1f} ms. rest time(+30tks) = {rest_time*1000:.2f} ms. prefill base time = {self.prefill_base_runtime(prompt_length)*1000:.2f} ms")

        time_rest = np.array(time_rest)


        bs, l, r = 0, 0, len(ontime_seq_groups)
        while l <= r:
            m = (l+r+1) // 2
            if m == 0:
                assert bs == 0
                logger.debug(f"m = 0, exit.")
                break
            pmp_timecost = pmp_timecosts[m-1] if m > 0 else 0
            logger.debug(f"switch P binary search: decode batch size = {len(decode_reqs)}, (l, r, m) = ({l}, {r}, {m}). pmp cost = {pmp_timecost} s.")
            time_rest_tmp = time_rest[:len_decode+m]-pmp_timecost
            logger.debug(f"time_rest now:{time_rest_tmp}")
            if self.check_batch_slo(time_rest=time_rest_tmp, prompt_lengths=prompt_lengths[:len_decode+m], output_lengths=output_lengths[:len_decode+m], predict_lengths=predict_lengths[:len_decode+m], stage='decode'):
                bs = max(bs, m)
                l = m + 1
                logger.debug(f"Search step succeed. pmp batch size -> {bs}")
            else:
                r = m - 1
        
        
        logger.debug(f"Decode req :{len(decode_reqs)},  len(ontime) = {len(ontime_seq_groups)}, len(timeout) = {len(timeout_seq_groups)}")

        if bs > 0:
            logger.info(f"DO Prefill. bs = {bs}")
            return True, bs, ontime_seq_groups, timeout_seq_groups
        else:
            logger.info(f"Evice Prefill.")
            return False, bs, ontime_seq_groups, timeout_seq_groups
        
        
    def check_seq_slo(self, output: RequestOutput):
        pmplen = len(output.prompt_token_ids)
        genlen = len(output.outputs[0].token_ids)
        except_ptime = output.pslo*self.prefill_base_runtime(pmplen)
        except_dtime = output.dslo*self.decode_stage_time_base(pmplen, genlen)
        slo = output.arrival_time + output.pslo*self.prefill_base_runtime(pmplen) + output.dslo*self.decode_stage_time_base(pmplen, genlen)

        return (genlen != 0) and (output.finished_time < slo), slo, except_ptime, except_dtime,  f"slo: {slo}. \nfin: {output.finished_time}"
    
    def check_seq_step_slo(self, output: RequestOutput, alpha: float = 0.99, end_time: Optional[float]=None, start_time: Optional[float]=None):
        '''
            检查seq中每一步的SLO是否满足
        '''
        pmplen = len(output.prompt_token_ids)
        genlen = len(output.outputs[0].token_ids)

        num_vio = 0
        num_attained = 0
        budget = output.budget
        e2e = [0,0,]
        if not output.step_time:
            return False, "Drop", genlen, 0, budget, e2e

        ptime = output.step_time[0] - output.arrival_time
        except_ptime = output.pslo*self.prefill_base_runtime(pmplen)

        if ptime <= except_ptime + budget:
            if (end_time is None or start_time is None) or (output.step_time[0] <= end_time and output.step_time[0] >= start_time):
                e2e[0] += 1
                num_attained += 1
        else:
            num_vio += 1
        num_step = 1
        for step in range(1, genlen):
            if output.step_time[step] == 0:
                num_step += 1
                continue
            dtime = output.step_time[step] - output.arrival_time
            except_dtime = output.dslo*self.decode_stage_time_base(pmplen, step + 1) + except_ptime + budget
            if dtime > except_dtime:
                num_vio += num_step
            else:
                if (end_time is None or start_time is None) or (output.step_time[step] <= end_time and output.step_time[step] >= start_time):
                    num_attained += num_step
            num_step = 1
        budget = except_dtime - dtime ## 应该直接看最后一个decode时间
        if dtime <= except_dtime: ## e2e来看decode是否超时
            if (end_time is None or start_time is None) or (output.step_time[step] <= end_time and output.step_time[step] >= start_time):
                e2e[1] += 1
        return True, "NULL MSG", num_vio, num_attained, budget, e2e

    def __repr__(self) -> str:
        return f"Llama length predictor with alpha = {self.alpha}"
    
def get_poisson_time(rate: float, total_time: float, seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)

    num_events = int(rate * total_time)  # 要模拟的事件总数
    inter_arrival_times = np.random.exponential(scale=1/rate, size=num_events)
    # arrival_times = np.cumsum(inter_arrival_times)
    return inter_arrival_times

class GammaProcess:
    def __init__(self, arrival_rate: float, cv: float) -> None:
        self.rate = arrival_rate
        self.cv = cv
        self.shape = 1/(cv**2)
        self.scale = cv * cv / arrival_rate

    def get_gamma_time(self, request_count: int, seed: int = 0):
        np.random.seed(seed)
        intervals = np.random.gamma(self.shape, self.scale, size=request_count)
        return intervals