from typing import List, Dict, Optional, Deque
import numpy as np


from vllm.logger import init_logger, add_consolehandler, add_filehandler
from vllm.sequence import SequenceGroup
from vllm.outputs import RequestOutput
import logging

logger = init_logger("Predictor")
logger.setLevel(logging.DEBUG)
add_consolehandler(logger, logging.INFO)
# add_filehandler(logger,"predictor.log", logging.DEBUG)
# add_filehandler(logger, f"all.log")
logger.info("Predictor Imported.")

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

class PredictorBase:
    length_predict = Dict[int, int]
    def _predict_len(self, prompt_len, generate_len) -> int:
        return self.length_predict[prompt_len+generate_len]
    
    # def predict_runtime(seq_groups: List[SequenceGroup], stage: str):
    #     raise NotImplementedError
    
class PredictorOutput:
    def __init__(
        self,
        do_prefill: bool,
        prefill_batch_size: int,
        ontime_seq_groups: List[SequenceGroup],
        timeout_seq_groups: List[SequenceGroup],
    ) -> None:
        self.do_prefill = do_prefill
        self.prefill_batch_size = prefill_batch_size
        self.ontime_seq_groups = ontime_seq_groups
        self.timeout_seq_groups = timeout_seq_groups
        

class Predictor(PredictorBase):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha
        self.length_predict = {i:33-i if i < 33 else 1 for i in range(0, 10000)}
        # self.length_predict = {i:1 for i in range(0, 4096)}
        self.lenmeta = 32


        # opt-13b tp=1 A6000
        # self.prefill_time = np.array(
        #     [0.04154373100027442, 0.04809862794354558, 0.04692623903974891, 0.048848932958208025, 0.06557490909472108, 0.061526434030383825, 0.06739221792668104, 0.06827137805521488, 0.12218902993481606, 0.11887189908884466, 0.09391803003381938, 0.09828791592735797, 0.1368049350567162, 0.1331000509671867, 0.13041201210580766, 0.13272114808205515, 0.15119397197850049, 0.15888213703874499, 0.17399779008701444, 0.16780850302893668, 0.1913573449710384, 0.1873393610585481, 0.19004243100062013, 0.19241312996018678, 0.23189778090454638, 0.23052372492384166, 0.23184060805942863, 0.2331338010262698, 0.24540892895311117, 0.24511416698805988, 0.26205060898792, 0.2624343930510804, 0.2912170479539782, 0.29374019999522716, 0.29757010703906417, 0.3077147639123723, 0.3331880480982363, 0.3436164240119979, 0.318013645010069, 0.3250786969438195, 0.36882328800857067, 0.3541405909927562, 0.3535566700156778, 0.3585680059622973, 0.3657867460278794, 0.3876180079532787, 0.39271058700978756, 0.39129543909803033, 0.43763504491653293, 0.4379435940645635, 0.4616922199493274, 0.46705836802721024, 0.47116873203776777, 0.4685066840611398, 0.4845849140547216, 0.5083939139731228, 0.5104001769796014, 0.5172322540311143, 0.484730553929694, 0.48665794800035655, 0.5131900240667164, 0.5214527579955757, 0.5244056229712442, 0.5187608749838546]
        # )
        # self.timemodle_para_p = (0.013511968077884007, 0.0002467961652387274, -2.800508238190365e-09)
        # self.timemodle_para_d = (0.04386336112795071, 1.0530766871910861e-06)

        # opt-30b tp=2 A6000
        # self.prefill_time = np.array(
        #     [0.055193306994624436, 0.05726578098256141, 0.06202950899023563, 0.07150365295819938, 0.08614647993817925, 0.09257684892509133, 0.09823908400721848, 0.10178257606457919, 0.1276996110100299, 0.13108079694211483, 0.13536463002674282, 0.13829364092089236, 0.18223482801113278, 0.18744655896443874, 0.19235880207270384, 0.19650455098599195, 0.2069451849674806, 0.20835728000383824, 0.24311186000704765, 0.24108051706571132, 0.2658948580501601, 0.27635430300142616, 0.2799296340672299, 0.27033198601566255, 0.3231101870769635, 0.3254572900477797, 0.33038321300409734, 0.3397328039864078, 0.34177550696767867, 0.3463383991038427, 0.38818667898885906, 0.3913901000050828, 0.3795004889834672, 0.3835976760601625, 0.3863373720087111, 0.41304646094795316, 0.4502584880683571, 0.45405276597011834, 0.45775585097726434, 0.4739716750336811, 0.49735395098105073, 0.5017526670126244, 0.5144925330532715, 0.5187127739191055, 0.4995893390150741, 0.529490509070456, 0.5335627250606194, 0.5363095100037754, 0.5889570260187611, 0.5923425509827211, 0.6169639149447903, 0.623993293964304, 0.6032062961021438, 0.6067754180403426, 0.6132854690076783, 0.6359887439757586, 0.6416862619807944, 0.6456154820043594, 0.6497032080078498, 0.653461245005019, 0.710882197949104, 0.7226511129410937, 0.7203858729917556, 0.7245263999793679]
        # )
        # self.timemodle_para_p = (0.027307203178189653, 0.0003323153299157181, 8.706391148549e-09)
        # self.timemodle_para_d = (0.05460089858371438, 9.136359951681938e-07)


        # opt-66b tp=4 A6000
        self.prefill_time = np.array(
            [0.07411340903490782, 0.07838616904336959, 0.08576463698409498, 0.09083482890855521, 0.120361662004143, 0.12998087401501834, 0.13483475998509675, 0.14159189991187304, 0.1912877910071984, 0.19752738601528108, 0.2047236500075087, 0.21100036602001637, 0.2677833669586107, 0.2736975740408525, 0.2799763409420848, 0.28583358600735664, 0.3133844720432535, 0.31295765994582325, 0.36498274689074606, 0.37184168002568185, 0.3929107260191813, 0.3988500719424337, 0.40609186701476574, 0.4092046240111813, 0.45941477094311267, 0.4860567409778014, 0.49400929699186236, 0.500126548926346, 0.4930105290841311, 0.4989168649772182, 0.5249348100041971, 0.534136800095439, 0.5422321490477771, 0.5471214660210535, 0.5539109279634431, 0.6146825030446053, 0.6226943300571293, 0.6299025430344045, 0.6370890960097313, 0.644656480057165, 0.6904278619913384, 0.6957412420306355, 0.7025084139313549, 0.7111638389760628, 0.7180413530441001, 0.7765017639612779, 0.7816034180577844, 0.7946464670822024, 0.8086638340028003, 0.8202158010099083, 0.862985547981225, 0.8734304850222543, 0.8765096520073712, 0.8831434410531074, 0.8979117280105129, 0.919158388976939, 0.961797283962369, 0.9693576289573684, 0.9703443580074236, 0.9780820780433714, 1.0197061840444803, 1.0297245290130377, 1.0322495509171858, 1.0424543810077012]
        )
        self.timemodle_para_p = (0.03330899765431316, 0.0004740191671711913, 2.5876625483425424e-08)
        self.timemodle_para_d = (0.07093051563015396, 7.632563111114172e-07)

        self.pred_step_time = 0.01
        self.pred_step = 100
        self.pred_step_p = 100
        # self.pred_gen_len = 80  #预测一个新的请求能生成多少token 

        self.decision_histoty: List[int] = []
        self.step_time_history: List[float] = []

        self.last_decision: int = 0
        # self.last_decode_time: float = self.timemodle_para_d[0] * 1.05
        # self.base_decode_time: float = self.timemodle_para_d[0] * 1.1
        self.last_decode_time: float = 0.072
        self.base_decode_time: float = 0.072

    def prefill_base_runtime(self, prompt_len: int) -> float:
        # 计算bs=1的时候，该长度的prompt的prefill阶段时间
        return self.prefill_time[(prompt_len+self.lenmeta-1)//self.lenmeta - 1]

    def decode_stage_time_base(self, prompt_len: int, gen_len: int):
        '''
        上一个函数去除循环的版本
        '''
        items = gen_len-1
        res = self.timemodle_para_d[0]*items + self.timemodle_para_d[1]*(items*(items+1)/2 + prompt_len*items)
        return res

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
            # 直接查表，不计算
            lens_sum = lengths.cumsum()
            ids = (lens_sum.astype(int) + 31) // 32 - 1
            mask = ids < 64
            res[mask] = self.prefill_time[ids[mask]]
            # return self.prefill_time[ids]
            return res
            # return res + 0.5 * (lengths.cumsum() < 250) * res
        elif stage == "decode":
            return self.timemodle_para_d[0] + self.timemodle_para_d[1]*lengths.cumsum()
        else:
            raise NotImplementedError

    def get_last_budget(self, dslo:float):
        return dslo*self.base_decode_time - self.last_decode_time

    def predict_len(self, seq_group: SequenceGroup) -> int:
        seq = seq_group.get_unfinished_seqs()[0]
        return self._predict_len(seq.get_prompt_len(), seq.get_output_len())

    def get_punctual_requests(
            self,
            seq_groups: Deque[SequenceGroup],
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

            rest_time = seq_group.arrival_time \
                        + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo \
                        - now \
                        + max(0, (self.pred_step_p - req.get_output_len())) * self.get_last_budget(seq_group.decode_slo)

            if rest_time < self.prefill_base_runtime(prompt_length):
                logger.debug(
                    f"Drop Request {seq_group.request_id} because it is timeout. "
                    f"rest time: {rest_time*1000:.1f} ms. "
                    f"base run time:{self.prefill_base_runtime(prompt_length)*1000:.1f} ms."
                )
                timeout_seq_groups.append(seq_group) 
                seq_groups.popleft()
            else:
                punctual_seq_groups.append(seq_group)
                seq_groups.popleft()
        
        return punctual_seq_groups, timeout_seq_groups

    def check_batch_slo(
            self,
            time_rest: List[float],
            prompt_lengths: List[int],
            output_lengths: List[int],
            predict_lengths: List[int],
            stage: str,
        ) -> bool:
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
                predict_lengths_np = np.array(predict_lengths)
                running_mask = (predict_lengths_np > 0)
                max_pred_len = predict_lengths_np.max().item()
                for step in range(1, max_pred_len+1):
                    l1 = lengths[running_mask]

                    time_rest -= self.predict_runtime(lengths=l1, stage=stage)
                    if any(time_rest[running_mask] < 0):
                        return False
                    running_mask = predict_lengths_np > step
                    lengths += 1
            else:
                raise NotImplementedError
            
            return True

    def switch_P(
        self,
        prefill_reqs: Deque[SequenceGroup],
        decode_reqs: Deque[SequenceGroup],
        now: float,
    ):
        if not prefill_reqs:
            return PredictorOutput(
                do_prefill=False,
                prefill_batch_size=0,
                ontime_seq_groups=[],
                timeout_seq_groups=[]
            )
        if not decode_reqs:
            return PredictorOutput(
                do_prefill=True,
                prefill_batch_size=len(prefill_reqs),
                ontime_seq_groups=list(prefill_reqs),
                timeout_seq_groups=[]
            )

        ontime_seq_groups, timeout_seq_groups = self.get_punctual_requests(prefill_reqs, now)
        pmp_lens = np.array([seq_group.get_unfinished_seqs()[0].get_prompt_len() for seq_group in ontime_seq_groups])
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

            rest_time = seq_group.arrival_time \
                        + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo \
                        + self.decode_stage_time_base(prompt_length, output_length+1) * seq_group.decode_slo \
                        - now \
                        + max(0, (self.pred_step - req.get_output_len())) * self.get_last_budget(seq_group.decode_slo)
                        
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
            rest_time = seq_group.arrival_time \
                        + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo \
                        + self.decode_stage_time_base(prompt_length, output_length+1) * seq_group.decode_slo \
                        - now \
                        + max(0, (self.pred_step_p - req.get_output_len())) * self.get_last_budget(seq_group.decode_slo)
            time_rest.append(rest_time)

            logger.debug(
                f"Seq {seq_group.request_id:>4}: "
                f"rest time:{(seq_group.arrival_time + self.prefill_base_runtime(prompt_length) * seq_group.prefill_slo - now)*1000:.1f} ms. "
                f"rest time(+30tks) = {rest_time*1000:.2f} ms. "
                f"prefill base time = {self.prefill_base_runtime(prompt_length)*1000:.2f} ms"
            )

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

            if self.check_batch_slo(
                time_rest=time_rest_tmp,
                prompt_lengths=prompt_lengths[:len_decode+m],
                output_lengths=output_lengths[:len_decode+m],
                predict_lengths=predict_lengths[:len_decode+m],
                stage='decode'
            ):
                bs = max(bs, m)
                l = m + 1
                logger.debug(f"Search step succeed. pmp batch size -> {bs}")
            else:
                r = m - 1
        
        logger.debug(f"Decode req :{len(decode_reqs)},  len(ontime) = {len(ontime_seq_groups)}, len(timeout) = {len(timeout_seq_groups)}")
        # logger.debug(f"time rest: {time_rest}, prefill_time: {pmp_timecost}")

        if bs > 0:
            logger.info(f"DO Prefill. batch size = {bs}")
            return PredictorOutput(
                do_prefill=True,
                prefill_batch_size=bs,
                ontime_seq_groups=ontime_seq_groups,
                timeout_seq_groups=timeout_seq_groups,
            )
        else:
            logger.info(f"Evict Prefill.")
            return PredictorOutput(
                do_prefill=False,
                prefill_batch_size=bs,
                ontime_seq_groups=ontime_seq_groups,
                timeout_seq_groups=timeout_seq_groups,
            )
        
    def check_seq_slo(self, output: RequestOutput):
        pmplen = len(output.prompt_token_ids)
        genlen = len(output.outputs[0].token_ids)
        except_ptime = output.pslo*self.prefill_base_runtime(pmplen)
        except_dtime = output.dslo*self.decode_stage_time_base(pmplen, genlen)
        slo = output.arrival_time + output.pslo*self.prefill_base_runtime(pmplen) + output.dslo*self.decode_stage_time_base(pmplen, genlen)

        return (genlen != 0) and (output.finished_time < slo), slo, except_ptime, except_dtime,  f"slo: {slo}. \nfin: {output.finished_time}"
    
    def check_seq_step_slo(self, output: RequestOutput, alpha: float = 0.99):
        '''
            检查seq中每一步的SLO是否满足
        '''
        pmplen = len(output.prompt_token_ids)
        genlen = len(output.outputs[0].token_ids)

        num_vio = 0
        num_attained = 0

        prefill_attained = True
        e2e_attained = True

        if not output.step_time:
            # timeout prefill req
            return False, "Drop", genlen, 0, False, False

        ptime = output.step_time[0] - output.arrival_time
        except_ptime = output.pslo*self.prefill_base_runtime(pmplen)
        if ptime > except_ptime:
            num_vio += 1
            prefill_attained = False
            # return False, "Prefill violated.", genlen, num_attained
            # return False, f"Prefill stage timeout. Expected use {except_ptime}, Actually {ptime}."
        
        num_attained += 1

        for step in range(1, genlen):
            dtime = output.step_time[step] - output.arrival_time
            except_dtime = output.dslo*self.decode_stage_time_base(pmplen, step+1) + except_ptime
            if dtime > except_dtime:
                num_vio += 1
                if step == genlen - 1:
                    e2e_attained = False
                # return False, f"The {step} th decode stage timeout.(when generating the {step+1} token.) Expected use {except_dtime}, Actually {dtime}."
            else:
                num_attained += 1
        # return True, "SLO Attained."
        return num_vio == 0, "NULL MSG", num_vio, num_attained, prefill_attained, e2e_attained

   