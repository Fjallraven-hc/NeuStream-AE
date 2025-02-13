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
    def __init__(self, stream_module_list, input_queue, output_queue, id, log_prefix, deploy_ready, extra_vae_safety_time, image_size, instant_profile_log, step_slo_scale: float, step_delta: float, **kwargs):
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
        self.DiT_XL_2_profile_latency = {
            #"DiT": {"1": 0.01972036121785641, "2": 0.018571753084659578, "3": 0.019397278666496275, "4": 0.024648231767117976, "5": 0.02942073369771242, "6": 0.0327542197778821, "7": 0.03522938949242234, "8": 0.046610258609056474, "9": 0.050186933171004054, "10": 0.05120321824401617}, 
            # below is batch latency with cudnn and tf32 enabled, but for fp16, tf32 doesn't matter?
            "DiT": {'1': 0.017604099575430156, '2': 0.017825833953917028, '3': 0.018477643452584742, '4': 0.02469173302873969, '5': 0.02942004717513919, '6': 0.03272720616683364, '7': 0.03535442834347487, '8': 0.04665777732804417, '9': 0.05022930987551808, '10': 0.051100462961941956, '11': 0.05788471630960703, '12': 0.059985206332057714, '13': 0.07121700321137905, '14': 0.06724760262295604, '15': 0.0794491237513721, '16': 0.0812934876307845, '17': 0.08649879134073854, '18': 0.09107603489607573, '19': 0.09463087095320225, '20': 0.09774496657773853},
            "vae": {'1': 0.016486729375230016, '2': 0.031214803343558466, '3': 0.04594287731188691, '4': 0.060670951280215366, '5': 0.07539902524854382, '6': 0.09012709921687226, '7': 0.10485517318520071, '8': 0.11958324715352917, '9': 0.13431132112185762, '10': 0.14903939509018607}
            #"vae": {"1": 0.017947486291329067, "2": 0.03042532696078221, "3": 0.04502915232442319, "4": 0.06113041636223594, "5": 0.0758535637985915, "6": 0.08969547070252398, "7": 0.10408120877109468, "8": 0.11871222325911125, "9": 0.13510771539683145, "10": 0.14904609449828665, "11": 0.16450104828303058, "12": 0.17836392776419718}
        }
        self.DiT_XL_2_instant_profile_latency = {}
        self.step_slo_scale = step_slo_scale
        self.step_delta = step_delta
        # one step slo, controlled by two values
        self.slo_batch_size = 0
        #print(f"yhc debug:: {type(self.DiT_XL_2_profile_latency["DiT"]["1"])}")
        # print(f"yhc debug:: step_slo_scale={type(step_slo_scale)}")
        # print(f"yhc debug:: step_delta={type(step_delta)}")
        if self.id == "DiT":
            self.batch_upper_bound = 20
            self.step_slo_latency = self.DiT_XL_2_profile_latency["DiT"]["1"] * self.step_slo_scale * self.step_delta
            while self.DiT_XL_2_profile_latency["DiT"][str(self.slo_batch_size + 1)] < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        elif self.id == "Vae":
            self.batch_upper_bound = 10
            self.step_slo_latency = self.DiT_XL_2_profile_latency["vae"]["1"] * self.step_slo_scale * self.step_delta
            while self.DiT_XL_2_profile_latency["vae"][str(self.slo_batch_size + 1)] < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, slo_batch_size = {self.slo_batch_size}")

        # vae latency near linear: 
        # latency = 0.01472807396832845 * batch_size + 0.0017586554069015659
        self.DiT_instant_profile = {}
        for batch_size in range(1, 41):
            self.DiT_instant_profile[batch_size] = {"count": 0, "val": 0}
        self.instant_profile_trust = 1000000000
        self.init_request_latency = 0.0015012567086766164
        self.cross_worker_transmission_time = 0.035 # 丢�个保守��，偏大
        # image_size set to 256
        self.deploy_ready = deploy_ready
        self.extra_vae_safety_time = extra_vae_safety_time

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
                    if self.id == "DiT":
                        f = open(self.instant_profile_log_file, "w")
                        json.dump(self.DiT_instant_profile, f)
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
                if self.id == "DiT":
                    if request["SLO"] + request["request_time"] < time.time() + self.DiT_XL_2_profile_latency["DiT"]["1"] * request["num-sampling-steps"] + self.cross_worker_transmission_time + self.DiT_XL_2_profile_latency["vae"]["1"]:
                        #print(f'left value: {item["SLO"] + item["request_time"]}')
                        #print(f'right value: {time.time() + self.DiT_XL_2_profile_latency["DiT"]["1"] * item["num-sampling-steps"] + self.DiT_XL_2_profile_latency["vae"]["1"]}')
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
                elif self.id == "Vae":
                    if request["SLO"] + request["request_time"] < time.time() + self.DiT_XL_2_profile_latency["vae"]["1"]:
                        #print(f'left value: {item["SLO"] + item["request_time"]}')
                        #print(f'right value: {time.time() + self.DiT_XL_2_profile_latency["vae"]["1"]}')
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        self.log_file.write(info+"\n")
                        #print(info)
                        #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request.")
                        #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, goodput rate: {self.goodput / self.total_request_count}, goodput: {self.goodput}, total_request: {self.total_request_count}")
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
                            if item.data["remain_loop_count_no_cuda"] < 0:
                                item.data[self.module_tag+"_send_time"] = time.time()
                                #print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, put one item to output queue.")
                                for key in item.data.keys():
                                    if type(item.data[key]) == torch.Tensor:
                                        item.data[key] = item.data[key].cpu()
                                print(f"DiT module put request {item.data["id"]} to output_queue")
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
            #self.log_file.flush()
            scatter()
            
            put_back_end = time.time()
            #print(f"yhc debug:: time used for put_back is {put_back_end - start}")

            # 将batch_queue里所有超时的request都扔掄1�7
            # 在DiT的case里，只有DiT霢�要这样做，Vae会早早扔掄1�7
            if self.id == "DiT":
                valid_request_list = []
                while not self.batch_queue.empty():
                    item = self.batch_queue.get()
                    # 使用instant_profile估计
                    if self.DiT_instant_profile[1]["count"] >= self.instant_profile_trust:
                        if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.DiT_instant_profile[1]["val"] * (item.data["remain_loop_count_no_cuda"] + 1) + self.cross_worker_transmission_time + self.DiT_XL_2_profile_latency["vae"]["1"]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                    # 使用预先profile估计
                    else:
                        if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.DiT_XL_2_profile_latency["DiT"]["1"] * (item.data["remain_loop_count_no_cuda"] + 1) + self.cross_worker_transmission_time + self.DiT_XL_2_profile_latency["vae"]["1"]:
                            # print(f'left value: {item[1]["SLO"] + item[1]["request_time"] - time.time()}')
                            # print(f'right value: {self.DiT_XL_2_profile_latency["DiT"]["1"] * (item[1]["remain_loop_count_no_cuda"] + 1) + self.DiT_XL_2_profile_latency["vae"]["1"]}')
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}, remain steps: {item.data["remain_loop_count_no_cuda"] + 1}"
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
                    if self.DiT_instant_profile[1]["count"] >= self.instant_profile_trust:
                        if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.DiT_instant_profile[1]["val"] * (item.data["remain_loop_count_no_cuda"] + 1) + self.cross_worker_transmission_time + self.DiT_XL_2_profile_latency["vae"]["1"]:
                            # print(f'left value: {item[1]["SLO"] + item[1]["request_time"] - time.time()}')
                            # print(f'right value: { + self.DiT_XL_2_profile_latency["DiT"]["1"] * (item[1]["remain_loop_count_no_cuda"] + 1) + self.DiT_XL_2_profile_latency["vae"]["1"]}')
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}, remain steps: {item.data["remain_loop_count_no_cuda"] + 1}"
                            self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                    # 使用预先profile估计
                    else:
                        if item.data["request_time"] + item.data["SLO"] - time.time() <=  self.DiT_XL_2_profile_latency["DiT"]["1"] * (item.data["remain_loop_count_no_cuda"] + 1) + self.cross_worker_transmission_time + self.DiT_XL_2_profile_latency["vae"]["1"]:
                            #print(f'left value: {item[1]["SLO"] + item[1]["request_time"]}')
                            #print(f'right value: {time.time() + self.DiT_XL_2_profile_latency["DiT"]["1"] * (item[1]["remain_loop_count_no_cuda"] + 1) + self.DiT_XL_2_profile_latency["vae"]["1"]}')
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                # 有效的request放回batch_queue
                for item in valid_request_list:
                    # 更新priority
                    # 不更新priority，避免出现交替schedule
                    #new_rest_time = item[1]["request_time"] + item[1]["SLO"] - time.time()
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
            #print(f"yhc debug:: time used for form batch is {end - abandon_timeout_end}")
            #print(f"yhc debug:: batch_size = {len(self.current_batch)}, time used for schedule batch is {end - start}")
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
        self.instant_profile_log_file = f"{self.log_prefix}_DiT_instant_profile_batch_latency.log"
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
                    if self.id == "DiT":
                        for _ in range(1):
                            batch_request = self.stream_module_list[0].compute(batch_request)
                            # update loop_index
                            for request in batch_request:
                                request["remain_loop_count"] -= 1
                                request["remain_loop_count_no_cuda"] -= 1
                    else:
                        for module in self.stream_module_list:
                            batch_request = module.compute(batch_request)
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
                    #if self.id == "DiT":
                        #self.DiT_instant_profile[batch_size]["val"] = (self.DiT_instant_profile[batch_size]["val"] * self.DiT_instant_profile[batch_size]["count"] + (end - begin)) / (self.DiT_instant_profile[batch_size]["count"] + 1)
                        #self.DiT_instant_profile[batch_size]["count"] += 1
                        #print(f"instant_profile: {self.DiT_instant_profile}")
                    self.batch_used.release()
                
            # Worker process receive interrupt
            except KeyboardInterrupt:
                print("-"*10, f"Worker process:[{os.getpid()}] received KeyboardInterrupt.", "-"*10)