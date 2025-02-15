import os
import time
import queue
import torch
import torch.multiprocessing as multiprocessing
import threading
import json

# PriorityQueue can not handle same value case, so need this class
class ComparableRequest:
    def __init__(self, priority, data):
        self.priority = priority
        self.data = data

    def __lt__(self, other):
        # Define comparison based on priority or any other logic
        return self.priority < other.priority

class Worker(multiprocessing.Process):
#class Worker(threading.Thread):
    def __init__(self, stream_module_list, input_queue, output_queue, id, log_prefix, deploy_ready, image_size, instant_profile_log, step_slo_scale: float, step_delta: float, **kwargs):
        super().__init__()
        self.stream_module_list = stream_module_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.id = id # mark worker kind
        self.batch_queue = queue.PriorityQueue() # (priority, item). lowest first
        self.low_priority_batch_queue = queue.PriorityQueue()
        self.current_batch = []
        #self.batch_ready = multiprocessing.Semaphore(0)
        #self.batch_used = multiprocessing.Semaphore(0)
        self.batch_ready = threading.Semaphore(0)
        self.batch_used = threading.Semaphore(0)
        self.first_batch = True
        self.loop_module_list = [type(stream_module).__name__ for stream_module in self.stream_module_list if stream_module.loop_module]
        module_name_list = [type(stream_module).__name__ for stream_module in self.stream_module_list]
        self.module_tag = "&".join(module_name_list)
        self.loop_unit = 1 if len(self.loop_module_list) != 0 else 1
        self.Palette_profile_latency = {
            "1": 0.018004728980362414,
            "2": 0.01952611024584621,
            "3": 0.024301754696294666,
            "4": 0.02958323527313769,
            "5": 0.0389538015909493,
            "6": 0.04583847851771861,
            "7": 0.05462223375029862,
            "8": 0.061911145448684694,
            "9": 0.07226610077451914,
            "10": 0.08018422306329012,
            "11": 0.0893126859376207
            }

        self.Palette_instant_profile_latency = {}
        self.step_slo_scale = step_slo_scale
        self.step_delta = step_delta
        # one step slo, controlled by two values
        self.slo_batch_size = 0
        #print(f"yhc debug:: {type(self.Palette_profile_latency["1"])}")
        # print(f"yhc debug:: step_slo_scale={type(step_slo_scale)}")
        # print(f"yhc debug:: step_delta={type(step_delta)}")
        self.batch_upper_bound = 11
        self.step_slo_latency = self.Palette_profile_latency["1"] * self.step_slo_scale * self.step_delta
        while self.Palette_profile_latency[str(self.slo_batch_size + 1)] < self.step_slo_latency:
            self.slo_batch_size += 1
            if self.slo_batch_size >= self.batch_upper_bound:
                break
        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, slo_batch_size = {self.slo_batch_size}")

        # vae latency near linear: 
        # latency = 0.01472807396832845 * batch_size + 0.0017586554069015659
        self.Palette_instant_profile = {}
        for batch_size in range(1, 12):
            self.Palette_instant_profile[batch_size] = {"count": 0, "val": 0}
        self.instant_profile_trust = 1000000000

        # image_size set to 256
        self.deploy_ready = deploy_ready

        self.total_request_count = 0
        self.goodput = 0
        self.log_prefix = log_prefix
        self.instant_profile_log = instant_profile_log
        if "device" in kwargs:
            self.device = kwargs["device"]
            for stream_module in self.stream_module_list:
                stream_module.device = self.device
        else:
            self.device = "cuda" # default to cuda:0
        print("-"*10, f"yhc debug:: loop_module_list: {self.loop_module_list}", "-"*10)

    def set_device(self, device, **kwargs):
        self.device = device

    def deploy(self, **kwargs):
        # deploy all stream_modules
        for stream_module in self.stream_module_list:
            stream_module.deploy()

    def gather(self, **kwargs):
        # determine whether terminate the process, 
        self.terminate_receive_flag = False
        self.terminate_schedule_flag = False
        #print(f"pid: [{os.getpid()}], scheduling batch thread working.")
        while True:
            # put request to batch_queue
            while not self.input_queue.empty():
                #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, get one item from input queue.")
                request = self.input_queue.get()
                if request == None:
                    #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, received terminate signal!")
                    self.output_queue.put(None)
                    self.terminate_receive_flag = True
                    if self.id == "Palette":
                        f = open(self.instant_profile_log_file, "w")
                        json.dump(self.Palette_instant_profile, f)
                    break

                # log the request info
                # 这种小数量的IO，时间代价应该在e-05秒，可忽略不讄1�7
                info = {
                    "request_time": request["request_time"],
                    "id": request["id"],
                    "SLO": request["SLO"]
                }
                self.log_file.write("receive request: "+json.dumps(info)+"\n")

                self.total_request_count += 1
                request[self.module_tag+"_receive_time"] = time.time()
                
                # check whether can serve
                if request["SLO"] + request["request_time"] < time.time() + self.Palette_profile_latency["1"] * request["num-sampling-steps"]:
                    #print(f'left value: {item["SLO"] + item["request_time"]}')
                    #print(f'right value: {time.time() + self.Palette_profile_latency["1"] * item["num-sampling-steps"] + self.Palette_S_2_profile_latency["vae"]["1"]}')
                    info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                    self.log_file.write(info+"\n")
                    #print(info)
                else:
                    #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, put one item from input queue to batch queue.")
                    # check tensor whether on same device
                    for key in request.keys():
                        if type(request[key]) == torch.Tensor:
                            request[key] = request[key].to(self.device)
                    # added in renewed v2
                    request["wait_loop_count"] = 0
                    self.batch_queue.put(ComparableRequest(time.time(), request))
                    #self.batch_queue.put((item["request_time"] + item["SLO"] - time.time(), item))
                    #self.batch_queue.put((item["request_time"] + item["SLO"] - time.time(), item))
                self.log_file.flush()
                        
            # stop working
            if self.batch_queue.qsize() == 0 and self.terminate_receive_flag and len(self.current_batch) == 0:
                self.terminate_schedule_flag = True
                # release lock, let running loop liberate from lock
                self.batch_ready.release()
                print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, terminate schedule!")
                break
                
            # empty queue, no need to form batch
            if self.batch_queue.qsize() == 0 and len(self.current_batch) == 0:
                continue

            # avoid concurrent access of self.current_batch
            if self.first_batch:
                self.first_batch = False
            else:
                self.batch_used.acquire()
                #print(f"{self.module_tag}, batch_used acquired once.")

            self.schedule_begin = time.time()

            def scatter():
                # put finished request to output_queue
                if len(self.current_batch) != 0:
                    if len(self.loop_module_list) != 0:
                        for item in self.current_batch:
                            # below is debug info
                            # print("-"*10, f"loop_index: {batch[1]['loop_index'][loop_module]}", "-"*10)
                            if item.data["remain_loop_count_no_cuda"] <= 0:
                                item.data[self.module_tag+"_send_time"] = time.time()
                                #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, put one item to output queue.")
                                for key in item.data.keys():
                                    if type(item.data[key]) == torch.Tensor:
                                        item.data[key] = item.data[key].cpu()
                                print(f"Palette module put request {item.data['id']} to output_queue")
                                self.output_queue.put(item.data)
                            else:
                                # 没跑够loop_count，重新回queue
                                # check whether current request has wait_loop_count
                                if item.data["wait_loop_count"] == 0:
                                    self.batch_queue.put(item)
                                else:
                                    self.low_priority_batch_queue.put(item)
                    # no loop module
                    else:
                        #self.total_request_count += len(self.current_batch)
                        for item in self.current_batch:
                            item.data[self.module_tag+"_send_time"] = time.time()
                            if time.time() <= item.data["SLO"] + item.data["request_time"]:
                                self.goodput += 1
                                for key in item.data.keys():
                                    if type(item.data[key]) == torch.Tensor:
                                        item.data[key] = item.data[key].cpu()
                                self.output_queue.put(item.data)
                            #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, put one item to output queue.")
                            # tensor have to be moved to cpu when passing across device
                        info = f"goodput: {self.goodput}, total_request: {self.total_request_count}\n"
                        self.log_file.write(info)
                        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, goodput rate: {self.goodput / self.total_request_count}, goodput: {self.goodput}, total_request: {self.total_request_count}")
            scatter()
            #self.log_file.flush()
            
            put_back_end = time.time()
            #print(f"yhc debug:: time used for put_back is {put_back_end - start}")

            # 将batch_queue里所有超时的request都扔掉
            valid_request_list = []
            while not self.batch_queue.empty():
                item = self.batch_queue.get()
                # 使用instant_profile估计
                if self.Palette_instant_profile[1]["count"] >= self.instant_profile_trust:
                    if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.Palette_instant_profile[1]["val"] * (item.data["remain_loop_count_no_cuda"]):
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                        self.log_file.write(info+"\n")
                        print(info)
                    else:
                        valid_request_list.append(item)
                # 使用预先profile估计
                else:
                    if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.Palette_profile_latency["1"] * (item.data["remain_loop_count_no_cuda"]):
                        # print(f'left value: {item[1]["SLO"] + item[1]["request_time"] - time.time()}')
                        # print(f'right value: {self.Palette_profile_latency["1"] * (item[1]["remain_loop_count_no_cuda"] + 1) + self.Palette_S_2_profile_latency["vae"]["1"]}')
                        info = f'pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data["id"]}, remain steps: {item.data["remain_loop_count_no_cuda"] + 1}'
                        self.log_file.write(info+"\n")
                        print(info)
                    else:
                        valid_request_list.append(item)
            # 有效的request放回batch_queue
            for item in valid_request_list:
                # 更新priority
                # 不更新priority，避免出现交替schedule
                #new_rest_time = item[1]["request_time"] + item[1]["SLO"] - time.time()
                self.batch_queue.put(item)
            
            valid_request_list = []
            while not self.low_priority_batch_queue.empty():
                item = self.low_priority_batch_queue.get()
                # 使用instant_profile估计
                if self.Palette_instant_profile[1]["count"] >= self.instant_profile_trust:
                    if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.Palette_instant_profile[1]["val"] * (item.data["remain_loop_count_no_cuda"]):
                        # print(f'left value: {item[1]["SLO"] + item[1]["request_time"] - time.time()}')
                        # print(f'right value: { + self.Palette_profile_latency["1"] * (item[1]["remain_loop_count_no_cuda"] + 1) + self.Palette_S_2_profile_latency["vae"]["1"]}')
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}, remain steps: {item.data['remain_loop_count_no_cuda'] + 1}"
                        self.log_file.write(info+"\n")
                        print(info)
                    else:
                        valid_request_list.append(item)
                # 使用预先profile估计
                else:
                    if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.Palette_profile_latency["1"] * (item.data["remain_loop_count_no_cuda"]):
                        #print(f'left value: {item[1]["SLO"] + item[1]["request_time"]}')
                        #print(f'right value: {time.time() + self.Palette_profile_latency["1"] * (item[1]["remain_loop_count_no_cuda"] + 1) + self.Palette_S_2_profile_latency["vae"]["1"]}')
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                        self.log_file.write(info+"\n")
                        print(info)
                    else:
                        valid_request_list.append(item)
            # 有效的request放回batch_queue
            for item in valid_request_list:
                self.low_priority_batch_queue.put(item)
            # 及时写回
            #self.log_file.flush()
            
            abandon_timeout_end = time.time()
            #print(f"yhc debug:: time used for abandon timeout is {abandon_timeout_end - put_back_end}")
            # form new batch
            self.current_batch = []
            
            self.high_priority_count = 0
            self.low_priority_count = 0

            # first put request with no wait_loop_count
            while self.batch_queue.qsize() > 0 and len(self.current_batch) < self.slo_batch_size:
                self.high_priority_count += 1
                self.current_batch.append(self.batch_queue.get())

            # put unscheduled request into low_priority queue
            while not self.batch_queue.empty():
                new_item = self.batch_queue.get()
                new_item.data["wait_loop_count"] += 1
                self.low_priority_batch_queue.put(ComparableRequest(new_item.data["wait_loop_count"], new_item.data))

            # if not exceed slo_batch_size, add low priority request
            while self.low_priority_batch_queue.qsize() > 0 and len(self.current_batch) < self.slo_batch_size:
                self.low_priority_count += 1
                self.current_batch.append(self.low_priority_batch_queue.get())

            # update the wait_loop_count in low_priority_queue
            temp_request_list = []
            while not self.low_priority_batch_queue.empty():
                request = self.low_priority_batch_queue.get().data
                request["wait_loop_count"] += 1
                temp_request_list.append(request)
            for request in temp_request_list:
                self.low_priority_batch_queue.put(ComparableRequest(request["wait_loop_count"], request))

            end = time.time()
            #print("-"*50)
            self.schedule_end = time.time()
            self.batch_ready.release()
            self.log_file.flush()

    @torch.no_grad()
    def run(self, **kwargs):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        log_file = open(f"{self.log_prefix}{self.module_tag}.log", "w")
        self.log_file = log_file
        self.instant_profile_log_file = f"{self.log_prefix}_Palette_instant_profile_batch_latency.log"
        torch.set_grad_enabled(False)
        with torch.inference_mode():
            try:
                #print("yhc test run.")
                print(f"pid: [{os.getpid()}], module list: {self.stream_module_list}")
                for module in self.stream_module_list:
                    module.device = self.device
                    module.deploy()
                    print(f"pid: [{os.getpid()}], serving module: {type(module).__name__}")
                self.deploy_ready.release()

                schedule_batch_thread = threading.Thread(target=self.gather)
                schedule_batch_thread.start()
                
                # sequentially run the batch module
                while True:
                    if self.terminate_schedule_flag == True:
                        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, terminate running!")
                        break
                    self.batch_ready.acquire()
                    #print(f"{self.module_tag}, batch_ready acquired once.")
                    if len(self.current_batch) == 0:
                        info = {
                            "batch_size": 0,
                            "time": time.time(),
                            "queue_size_before_schedule": 0,
                            "msg": "emptyqueue"
                        }
                        log_file.write(json.dumps(info) + "\n")
                        self.batch_used.release()
                        continue
                    
                    execution_begin = time.perf_counter()
                    
                    # wipe off the priority
                    # below list just passed reference, not data copy
                    batch_size = len(self.current_batch)
                    batch_request = [item.data for item in self.current_batch]
                    # execute through the pipeline
                    for _ in range(1):
                        batch_request = self.stream_module_list[0].compute(batch_request)
                        # update loop_index
                        for request in batch_request:
                            request["remain_loop_count_no_cuda"] -= 1

                    info = {
                        "batch_size": len(self.current_batch),
                        "time": time.time(),
                        "schedule_time": self.schedule_end - self.schedule_begin,
                        "execution_time": time.perf_counter() - execution_begin,
                        "high_priority_count": self.high_priority_count,
                        "low_priority_count": self.low_priority_count,
                        "queue_size_before_schedule": self.batch_queue.qsize() + len(self.current_batch) + self.low_priority_batch_queue.qsize(),
                        "batch_size_after_schedule": self.batch_queue.qsize(),
                        "running_requests_id_list": [item.data["id"] for item in self.current_batch],
                        "rest_time": [(item.data["request_time"] + item.data["SLO"] - time.time()) for item in self.current_batch]
                    }
                    log_file.write(json.dumps(info) + "\n") 

                    #if self.id == "Palette":
                        #self.Palette_instant_profile[batch_size]["val"] = (self.Palette_instant_profile[batch_size]["val"] * self.Palette_instant_profile[batch_size]["count"] + (end - begin)) / (self.Palette_instant_profile[batch_size]["count"] + 1)
                        #self.Palette_instant_profile[batch_size]["count"] += 1
                        #print(f"instant_profile: {self.Palette_instant_profile}")
                    self.batch_used.release()
                
            # Worker process receive interrupt
            except KeyboardInterrupt:
                print("-"*10, f"Worker process:[{os.getpid()}] received KeyboardInterrupt.", "-"*10)