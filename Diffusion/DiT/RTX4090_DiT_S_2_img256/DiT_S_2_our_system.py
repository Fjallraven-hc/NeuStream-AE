import copy
import uuid
import time
import torch
import numpy
import PIL
import json
from stream_module_list import DiTModule, VaeModule
from utils import *

def handle_output(output_queue, log_name, workload_request_count, rate, cv, slo_factor, trace_time):
    #import numpy
    #from PIL import Image
    from datetime import datetime, timezone, timedelta
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

    goodput_request_count = 0 # not include warm-up request
    request_count = 0

    f = open(log_name+timestamp, "w")

    output_queue.get()
    request_count += 1
    from torchvision.utils import save_image

    while True:
        if output_queue.qsize() == 0:
            time.sleep(0.01)
            continue
        result = output_queue.get()
        # stop the process
        if result == None:
            f.close()
            statistics = f"time:{timestamp}, DiT_S_2 256x image, rate:{rate} qps, cv={cv}, slo={slo_factor}, NeuStream goodput_rate={goodput_request_count}/{workload_request_count}, goodput speed={goodput_request_count / trace_time}\n"
            print(statistics)
            result_file = open("DiT_S_2_serve_result.txt", "a")
            result_file.write(statistics)
            break
        request_count += 1
        #print(f"handle request_count: {request_count}")
        
        # whether log the warm-up request
        if result["id"] < 0:
            continue
        
        goodput_request_count += 1
        #print("-"*20)
        f.write("-"*20+"\n")
        result["finish_time"] = time.time()
        f.write(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}\n")
        #print(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}")
        f.write(f'request step: {result["num-sampling-steps"]}')
        for key in result.keys():
            if type(result[key]) == float or key == "id" or type(result[key]) == int:
                f.write(f"key: {key}, value: {result[key]}\n")
                #print(f"key: {key}, value: {result[key]}")
        f.write(f"free time = {result["request_time"] + result["SLO"] - result["finish_time"]}")
        #save_image(torch.tensor(result["image_numpy_ndarray"]), f"{result["id"]}.png", normalize=True)
        #img = PIL.Image.fromarray(numpy.array(result["pillow_image"]).astype(numpy.uint8))
        #img.save(f"{result['id']}.jpg")
        #print("-"*20)
        f.write(f"collector worker receive workload request count = {goodput_request_count}\n")
        print(f"collector worker receive workload request count = {goodput_request_count}")
        f.flush()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="parameter for server.")

    parser.add_argument('--image_size', required=True, type=int, help='a value to determine image size')
    parser.add_argument('--rate_scale', required=True, type=str, help='a value to determine arrival rate')
    parser.add_argument('--cv_scale', required=True, type=str, help='a value to determine arrival coefficient of variation')
    parser.add_argument('--slo_scale', required=True, type=float, help='a value to determine slo factor')
    parser.add_argument('--extra_vae_safety_time', required=True, type=float, help='a value to determine extra budget for vae and safety')
    parser.add_argument('--log_folder', required=True, type=str, help='a value to determine log folder')
    parser.add_argument('--profile_device', required=True, type=str, help='a value to determine profile device')
    parser.add_argument('--run_device', required=True, type=str, help='a value to determine running device')
    parser.add_argument('--step_delta', required=True, type=float, help='a value to determine running device')

    args = parser.parse_args()

    gamma_process_trace_rtx4090_256 = json.load(open("DiT_S2_256_trace.json"))

    rate = args.rate_scale

    slo_factor = args.slo_scale

    image_size = args.image_size   

    cv = args.cv_scale

    extra_vae_safety_time = args.extra_vae_safety_time

    log_folder = args.log_folder

    device = args.profile_device

    run_device = args.run_device

    step_delta = args.step_delta

    key = f"rate={rate}_cv={cv}_{image_size}"

    # no need to pass gradient
    torch.set_grad_enabled(False)
    try:
        workload_request_count = 500
        time_pattern = "Gamma"

        from datetime import datetime, timezone, timedelta
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        log_prefix = f"{log_folder}/{timestamp}_{time_pattern}_rate={rate}_cv={cv}_slo_factor={slo_factor}_request={workload_request_count}_step_delta={step_delta}_device={args.profile_device}_image_size={image_size}_"
        instant_profile_log = f"{log_folder}/{timestamp}_request={workload_request_count}_instant_step_latency.log"

        # init queue
        worker_nums = 2

        input_queue = torch.multiprocessing.Manager().Queue()
        output_queue = torch.multiprocessing.Manager().Queue()
        queue_list = [torch.multiprocessing.Manager().Queue() for _ in range(worker_nums - 1)]

        deploy_ready = torch.multiprocessing.Semaphore(0)
        
        queue_list.insert(0, input_queue)
        queue_list.append(output_queue)  

        device = run_device
        DiT_module = DiTModule(device=device, data_type="float16", parameter_path="../DiT_pretrained_models/DiT-S-2-256x256.pt", DiT_config={})
        vae_module = VaeModule(device=device, data_type="float16", parameter_path="../sd-vae-ft-mse")

        # 创建工作进程
        worker_list = []
        worker_list.append(Worker(stream_module_list=[DiT_module], input_queue=queue_list[0], output_queue=queue_list[1], id="DiT", log_prefix=log_prefix, deploy_ready=deploy_ready, extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, instant_profile_log=instant_profile_log, step_slo_scale=slo_factor,  step_delta=step_delta))
        worker_list.append(Worker(stream_module_list=[vae_module], input_queue=queue_list[1], output_queue=queue_list[2], id="Vae", log_prefix=log_prefix, deploy_ready=deploy_ready, extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, instant_profile_log=instant_profile_log, step_slo_scale=slo_factor,  step_delta=step_delta))
    
        arrival_interval_list = gamma_process_trace_rtx4090_256[f"rate={rate},cv={cv}"]
        trace_time = sum(arrival_interval_list[:workload_request_count])/1.2
        collect_worker = torch.multiprocessing.Process(target=handle_output, args=(output_queue, log_prefix, workload_request_count, rate, cv, slo_factor, trace_time))
        collect_worker.start()

        deploy_begin = time.time()
        for _worker in worker_list:
            _worker.start()
        
        # wait for all module ready
        for _ in range(len(worker_list)):
            deploy_ready.acquire()
        print(f"Workers deploy all done! time used: {time.time() - deploy_begin}")
        
        DiT_S_2_profile_latency = {
            # 512 data: "DiT": {'1': 0.00853206742182374, '2': 0.009045315541326999, '3': 0.009520812202244997, '4': 0.009978152710944415, '5': 0.01158508762717247, '6': 0.01402720705792308, '7': 0.01590313234180212, '8': 0.0177910624332726, '9': 0.019693216670304537, '10': 0.021479962829500435, '11': 0.024451364584267138, '12': 0.026667664751410483, '13': 0.028740133870393036, '14': 0.03059479133784771, '15': 0.0327286289036274, '16': 0.03406328016519546, '17': 0.03680084356665611, '18': 0.038923687141388656, '19': 0.04112765133753419, '20': 0.043587814401835207},
            "DiT": {'1': 0.008644959650933743, '2': 0.00917433289065957, '3': 0.00959276306629181, '4': 0.009841575443744659, '5': 0.010188856922090053, '6': 0.010417315188795328, '7': 0.010932176448404789, '8': 0.011312654327601195, '9': 0.011764868374913931, '10': 0.012182640433311463, '11': 0.012500259295105934, '12': 0.01281594754382968, '13': 0.01318560914695263, '14': 0.013785529989749194, '15': 0.014103660233318806, '16': 0.014398379147052765, '17': 0.0146422200165689, '18': 0.01509671126678586, '19': 0.015466325741261245, '20': 0.01580424864217639},
            # 512 data: "vae": {'1': 0.03710936324670911, '2': 0.07858553188852965, '3': 0.11575526400469244, '4': 0.15218577816151083, '5': 0.1886397102009505, '6': 0.2247909987065941, '7': 0.2610476689506322, '8': 0.29755701271817087, '9': 0.3341629472654313}
            "vae": {'1': 0.010922675570473075, '2': 0.018185735922306778, '3': 0.026859085345640778, '4': 0.03519219818525016, '5': 0.0445969854388386, '6': 0.05348503946326673, '7': 0.0618163659889251, '8': 0.0702100186329335, '9': 0.07968861000612378, '10': 0.0884431685321033}
        }
        init_request_latency = 0.0015012567086766164

        warm_up_request = {
            "image_size": 256,
            "num-sampling-steps": 250,
            "remain_loop_count": 249,
            "remain_loop_count_no_cuda": 249,
            "class_label": 100,
            "cfg_scale": 4.0,
            "uuid": uuid.uuid1(),
            "request_time": time.time(),
            "SLO": slo_factor * (init_request_latency
                                + DiT_S_2_profile_latency["DiT"]["1"] * 250
                                + DiT_S_2_profile_latency["vae"]["1"]
                                ),
            "id": -1 # for debug
            }
        # warm up all the unet batch_size
        test_count = 15
        for idx in range(test_count):
            #warm_up_request["id"] = idx
            time.sleep(0.5)
            temp_request = copy.deepcopy(warm_up_request)
            temp_request["request_time"] = time.time()
            temp_request["id"] = -1 - idx
            input_queue.put(temp_request)
        # warm up all the unet batch_size

        time.sleep(20)
        print("warm up succeed!")
        
        num_sampling_steps_list = gamma_process_trace_rtx4090_256["random_num_sampling_steps"]
        class_label_list = gamma_process_trace_rtx4090_256["random_class_label"]

        for idx in range(workload_request_count):
            # time.sleep(5)
            time.sleep(arrival_interval_list[idx])
            input = {
                "image_size": 256,
                "num-sampling-steps": num_sampling_steps_list[idx],
                "remain_loop_count": num_sampling_steps_list[idx] - 1,
                "remain_loop_count_no_cuda": num_sampling_steps_list[idx] - 1,
                "class_label": class_label_list[idx],
                "cfg_scale": 4.0,
                "uuid": uuid.uuid1(),
                "request_time": time.time(),
                "SLO": slo_factor * (init_request_latency
                                     + DiT_S_2_profile_latency["DiT"]["1"] * num_sampling_steps_list[idx]
                                     + DiT_S_2_profile_latency["vae"]["1"]
                                     ),
                "id": idx # for debug
            }
            input_queue.put(input)
            print(f"input_queue put request id: {idx}""\n""-----------------------")
        # end request
        time.sleep(10) # wait for all done
        input_queue.put(None)
        
        for _worker in worker_list:
            _worker.join()

    except KeyboardInterrupt:
        print("-"*10, "Main process received KeyboardInterrupt","-"*10)