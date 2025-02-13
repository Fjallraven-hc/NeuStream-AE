"""
ClockWork 

"""
from dataclasses import dataclass
import copy
import uuid
import json
import time
import multiprocessing
from collections import deque
import numba
#from diffusers import StableDiffusionPipeline,EulerAncestralDiscreteScheduler
from DiT_S_2_pipe_for_clockwork import DiT_pipe
import torch
import argparse

# 512 data: DiT_info = {'1': 0.00853206742182374, '2': 0.009045315541326999, '3': 0.009520812202244997, '4': 0.009978152710944415, '5': 0.01158508762717247, '6': 0.01402720705792308, '7': 0.01590313234180212, '8': 0.0177910624332726, '9': 0.019693216670304537, '10': 0.021479962829500435, '11': 0.024451364584267138, '12': 0.026667664751410483, '13': 0.028740133870393036, '14': 0.03059479133784771, '15': 0.0327286289036274, '16': 0.03406328016519546, '17': 0.03680084356665611, '18': 0.038923687141388656, '19': 0.04112765133753419, '20': 0.043587814401835207}
DiT_info  = {'1': 0.008644959650933743, '2': 0.00917433289065957, '3': 0.00959276306629181, '4': 0.009841575443744659, '5': 0.010188856922090053, '6': 0.010417315188795328, '7': 0.010932176448404789, '8': 0.011312654327601195, '9': 0.011764868374913931, '10': 0.012182640433311463, '11': 0.012500259295105934, '12': 0.01281594754382968, '13': 0.01318560914695263, '14': 0.013785529989749194, '15': 0.014103660233318806, '16': 0.014398379147052765, '17': 0.0146422200165689, '18': 0.01509671126678586, '19': 0.015466325741261245, '20': 0.01580424864217639}
# 512 data: vae_info = {'1': 0.03710936324670911, '2': 0.07858553188852965, '3': 0.11575526400469244, '4': 0.15218577816151083, '5': 0.1886397102009505, '6': 0.2247909987065941, '7': 0.2610476689506322, '8': 0.29755701271817087, '9': 0.3341629472654313}
vae_info = {'1': 0.010922675570473075, '2': 0.018185735922306778, '3': 0.026859085345640778, '4': 0.03519219818525016, '5': 0.0445969854388386, '6': 0.05348503946326673, '7': 0.0618163659889251, '8': 0.0702100186329335, '9': 0.07968861000612378, '10': 0.0884431685321033}

pro = {}
for steps in range(200, 251):
    pro[steps] = {}
    for batch_size in range(1, 11):
        pro[steps][batch_size] = DiT_info[str(batch_size)] * steps + vae_info[str(batch_size)]

@dataclass(frozen=True)
class Strategy:
    steps: int
    batch_size: int
    latest: float

@dataclass(frozen=True)
class Work:
    id: int
    duration: float
    work_begin: float

class WorkerTracker:
    def __init__(self,track_queue,shm):
        self.track_queue = track_queue
        self.work_begin = 0
        self.lag = 0.01
        self.future = 0.001
        self.outstanding = deque()
        self.total_outstanding = 0
        self.shm = shm
    
    #@numba.jit
    def _update(self, id: int, time_of_completion: float):
        if(self.outstanding[0].id == id):
            self.total_outstanding -= self.outstanding[0].duration
            self.work_begin = time_of_completion
            self.outstanding.popleft()
        else:
            for ow in self.outstanding:
                if ow.id == id:
                    self.total_outstanding -= ow.duration
                    self.work_begin += ow.duration
                    self.outstanding.remove(ow)
                    break
        if self.outstanding:
            self.work_begin = max(self.work_begin, self.outstanding[0].work_begin)

    def available(self):
        now = time.time()
        work_begin = self.work_begin
        if self.outstanding and (work_begin+self.outstanding[0].duration + self.lag) < now:
            work_begin = now - self.lag - self.outstanding[0].duration
        return max(work_begin + self.total_outstanding, now + self.future)
    
    def add(self, id, duration, work_begin = 0):
        if not self.outstanding:
            self.work_begin = max(time.time(),self.work_begin)
        self.outstanding.append(Work(id, duration, work_begin))
        self.total_outstanding += duration
        return 0
    
    def success(self,id, time_of_completion):
        self._update(id,time_of_completion)

    def track(self):
        if not self.track_queue.empty():
            item = self.track_queue.get()
            self.success(item["id"],item["time_of_completion"])
        if self.shm[0] == -1:
            self.shm[0] = self.available()             
        if self.shm[1] == -1:
            self.shm[1] = self.add(self.shm[2],self.shm[3])

def worker_tracker(tracker_queue, shm):
    tracker = WorkerTracker(tracker_queue,shm)
    while True:
        tracker.track()

class Controller:
    def __init__(self,request_queue,infer_queue,shared_mem,instance,batchsize):
        self.instance = instance 
        self.batchsize = batchsize 
        self.strategies = []
        self.strategy = { i:[] for i in self.instance}
        self.sd = { i:{ j:deque() for j in self.batchsize }for i in self.instance }
        self.request_queue = request_queue
        self.infer_queue = infer_queue
        self.tracker = shared_mem
        self.schedule_ahead = 0.01
        self.id = 1

    #@numba.jit
    def pull_incoming_requests(self):
        size = self.request_queue.qsize() 
        for _ in range(size):
            item = self.request_queue.get()
            key = item["num-sampling-steps"]
            for b in self.batchsize:
                self.sd[key][b].append(item)
    
    #@numba.jit
    def add_model_strategy(self, instance: int, max_batch_size = None):
        if max_batch_size is None:
            max_batch_size = self.batchsize[-1]
        self.pull_incoming_requests()
        sta = []
        for b in reversed(self.batchsize):
            if len(self.sd[instance][b]) == 0 or b > max_batch_size:
                continue
            if len(self.sd[instance][b]) >= b:
                ddl = self.sd[instance][b][0]["request_time"] + self.sd[instance][b][0]["SLOs"]
                priority = ddl - pro[instance][b] 
                sta.append(Strategy(instance,b, priority))
        self.strategy[instance] = sta
        if not self.strategy[instance]:
            return
        for i in self.strategy[instance]:
            self.strategies.append(i)

    #@numba.jit
    def try_deque(self, free_at: float,instance: int, batch_size: int):
        self.pull_incoming_requests()
        for b in self.batchsize:
            exec_time = pro[instance][b] 
            complete_time = free_at + exec_time
            while(len(self.sd[instance][b]) > 0):
                item = self.sd[instance][b][0]
                if item["request_time"] + item["SLOs"] < complete_time:
                    self.sd[instance][b].popleft()
                else:
                    break
        i = 1
        while(i < len(self.sd[instance]) and len(self.sd[instance][i+1]) >= i+1 ):
            i += 1
        qe = self.sd[instance][i]
        if len(qe) < i or i < batch_size:
            return None
        
        input = []

        for _ in range(i):
            req = qe.popleft()
            input.append(req)
            seqno = req["id"]
        for q in self.sd[instance].values():
            while len(q) > 0 and q[0]["id"] <= seqno:
                q.popleft()
        return input
    
    def schedule_info(self, gpu_id):
        for i in self.instance:
            self.add_model_strategy(i)        
        if not self.strategies:
            return 
        while (self.strategies):
            self.tracker[gpu_id][0] = -1
            while(self.tracker[gpu_id][0] == -1):
                continue
            exe = self.tracker[gpu_id][0]
            self.tracker[gpu_id][0] = 0
            if time.time() + self.schedule_ahead < exe and exe != 0:
                break
            self.strategies.sort(key=lambda x: x.latest,reverse=True)
            strategy = self.strategies.pop()
            instance = strategy.steps
            self.strategies = [x for x in self.strategies if x.steps != instance ]
            self.strategy[instance] = []
            input = self.try_deque(exe, instance, strategy.batch_size)
            if input is None:
                self.add_model_strategy(instance, strategy.batch_size - 1)
            else:
                input[0]["track_id"] = self.id
                duration = pro[instance][len(input)] 
                self.tracker[gpu_id][2] = self.id
                self.tracker[gpu_id][3] = duration
                self.tracker[gpu_id][1] = -1 
                while (self.tracker[gpu_id][1] == -1):
                    continue                             
                self.infer_queue[gpu_id].put(input)
                self.id += 1
                break
        self.strategies = []

def controller_(request_queue,infer_queue,tracker_queue,num_worker,instance,batchsize):
    contr = Controller(request_queue,infer_queue,tracker_queue,instance,batchsize)
    i = 0
    while True:
        contr.schedule_info(i)
        i = (i+1)%num_worker

def infer_worker(infer_queue,pipe,idxx,tracker_queue, log_file):
    f = open(log_file, "w")
    data = {
        "good": 0,
        "finish": 0,
    }
    
    log = False
    #pipe("a girl",num_inference_steps=30)
    #pipe("a girl",num_inference_steps=35)
    pipe([100], 200, 3.0)
    pipe([100], 250, 4.0)
    
    @torch.inference_mode()
    #@numba.jit
    def infer(item, tracker_queue,pipe):
        good = 0
        steps = item[0]["num-sampling-steps"]
        cfg_scale = item[0]["cfg_scale"]
        track_id = item[0]["track_id"]
        batch = len(item)
        class_label_list = []
        for i in item:
            class_label_list.append(i["class_label"])
        #h = item[0]["height"]
        #w=h
        pipe(class_label_list, steps, cfg_scale)
        ed = time.time()
        for i in item:
            if ed <= i["request_time"] + i["SLOs"]:
                good += 1
        tracker_queue.put({"id":track_id,"time_of_completion": time.time()}) 
        sid = ""
        for i in item:
            sid += str(i["id"]) + " "
        return batch, good, sid
    
    while True:
        item = infer_queue.get()
        batch, good_, sid = infer(item, tracker_queue, pipe)
        if not log:
            log = not item[0]["warmup"]
        if log:
            data["good"] += good_
            data["finish"] += batch  
        info = f"{idxx}号worker ---- good : {data['good']} --- finish: {data['finish']} --- id: {sid} \n"
        f.write(info) 
        f.flush()
        print(info)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parameter for server.")

    parser.add_argument('--image_size', required=True, type=int, help='a value to determine image size')
    parser.add_argument('--rate_scale', required=True, type=str, help='a value to determine arrival rate')
    parser.add_argument('--cv_scale', required=True, type=str, help='a value to determine arrival coefficient of variation')
    parser.add_argument('--slo_scale', required=True, type=float, help='a value to determine slo factor')
    parser.add_argument('--log_folder', required=True, type=str, help='a value to determine log folder')
    parser.add_argument('--run_device', required=True, type=str, help='a value to determine running device')

    torch.multiprocessing.set_start_method("spawn")

    args = parser.parse_args()
    rate = args.rate_scale
    slo_factor = args.slo_scale
    cv = args.cv_scale
    log_folder = args.log_folder
    device = args.run_device

    test_request_count = 500 # 200
    from datetime import datetime, timezone, timedelta
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")    
    log_file_path = f"{log_folder}/{timestamp}_Gamma_rate={rate}_cv={cv}_slo_factor={slo_factor}_request={test_request_count}_device=RTX4090_image_size={args.image_size}.log"

    pipes = [        
        DiT_pipe(device),
        # StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:2"),
        # StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:4"),
        # StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:7"),
    ]

    instance = range(200, 251)
    batchsize = range(1,10)    
    num_worker = 1

    infer_queue = [multiprocessing.Queue() for _ in range(num_worker)]
    request_queue = multiprocessing.Queue()
    tracker_queue = [multiprocessing.Queue() for _ in range(num_worker) ]

    shared_mem = [multiprocessing.Array('d',[0,0,0,0]) for _ in range(num_worker)]

    processes = [multiprocessing.Process(target=infer_worker, args=(infer_queue[i],pipes[i],i,tracker_queue[i],log_file_path)) for i in range(num_worker)] + [
        multiprocessing.Process(target=controller_, args=(request_queue,infer_queue,shared_mem,num_worker,instance,batchsize)),
    ] + [multiprocessing.Process(target=worker_tracker,args=(tracker_queue[i],shared_mem[i])) for i in range(num_worker)]

    for p in processes:
        p.start()

    print("Successfully!")
    time.sleep(6)
    print("begin infer.")

    ## warm up
    warm_up_request = {
            "image_size": 256,
            "num-sampling-steps": 250,
            "class_label": 100,
            "cfg_scale": 4.0,
            "uuid": uuid.uuid1(),
            "request_time": time.time(),
            "SLOs": 5,
            "id": -1 ,# for debug
            "warmup": True,
            }
    for idx in range(10):
        time.sleep(0.3)
        temp_request = copy.deepcopy(warm_up_request)
        temp_request["request_time"] = time.time()
        temp_request["id"] = -1 - idx
        request_queue.put(warm_up_request)
    
    # wait for warm-up end
    time.sleep(15)

    gamma_process_trace_rtx4090_256 = json.load(open("DiT_S2_256_trace.json"))

    num_sampling_steps_list = gamma_process_trace_rtx4090_256["random_num_sampling_steps"]
    class_label_list = gamma_process_trace_rtx4090_256["random_class_label"]
    arrival_interval_list = gamma_process_trace_rtx4090_256[f"rate={rate},cv={cv}"]
    
    init_request_latency = 0.0015012567086766164

    ## measure
    begin = time.time()    
    for idx in range(test_request_count):
        time.sleep(arrival_interval_list[idx])
        # time.sleep(0.5)
        request = {
            "image_size": 256,
            "num-sampling-steps": num_sampling_steps_list[idx],
            "class_label": class_label_list[idx],
            "cfg_scale": 4.0,
            "uuid": uuid.uuid1(),
            "request_time": time.time(),
            "SLOs": slo_factor * (init_request_latency
                                    + DiT_info["1"] * num_sampling_steps_list[idx]
                                    + vae_info["1"]
                                    ),
            "id": idx ,# for debug
            "warmup": False,
        }
        request_queue.put(request)
    
    time.sleep(20)
    for p in processes:
        p.terminate()

    print("end infer.")
    for p in processes:
        p.join()
    print("throughput:", test_request_count / (time.time() - begin))
