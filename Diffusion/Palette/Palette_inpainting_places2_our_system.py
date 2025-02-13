import copy
import uuid
import time
import torch
import numpy
import PIL
import json
from stream_module_list import UNetModule
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
    
    while True:
        if output_queue.qsize() == 0:
            time.sleep(0.01)
            continue
        result = output_queue.get()
    
        # stop the process
        if result == None:
            f.close()
            statistics = f"time:{timestamp}, Palette 256x image, rate:{rate} qps, cv={cv}, slo={slo_factor}, NeuStream goodput_rate={goodput_request_count}/{workload_request_count}, goodput speed={goodput_request_count / trace_time}\n"
            print(statistics)
            result_file = open("Palette_serve_result.txt", "a")
            result_file.write(statistics)
            break
        request_count += 1
        
        # whether log the warm-up request
        if result["id"] < 0:
            continue
        
        goodput_request_count += 1
        f.write("-"*20+"\n")
        result["finish_time"] = time.time()
        f.write(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}\n")
        #print(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}")
        f.write(f'request step: {result["num-sampling-steps"]}\n')
        computation_time = 0
        for key in result.keys():
            if "receive_time" in key:
                computation_time -= result[key]
            if "send_time" in key:
                computation_time += result[key]
            if type(result[key]) == float or key == "id" or type(result[key]) == int:
                f.write(f"key: {key}, value: {result[key]}\n")
                #print(f"key: {key}, value: {result[key]}")
        transmission_time = result['finish_time'] - result['request_time'] - computation_time
        f.write(f'intra module time = {computation_time}\n')
        f.write(f'transmission time = {transmission_time}\n')
        f.write(f'free time = {result["request_time"] + result["SLO"] - result["finish_time"]}\n')
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
    parser.add_argument('--log_folder', required=True, type=str, help='a value to determine log folder')
    parser.add_argument('--profile_device', required=True, type=str, help='a value to determine profile device')
    parser.add_argument('--run_device', required=True, type=str, help='a value to determine running device')
    parser.add_argument('--step_delta', required=True, type=float, help='a value to determine running device')

    args = parser.parse_args()

    gamma_process_trace_rtx4090_256 = json.load(open("Palette_inpainting_places2_trace.json"))

    rate = args.rate_scale

    slo_factor = args.slo_scale

    image_size = args.image_size   

    cv = args.cv_scale

    log_folder = args.log_folder

    device = args.profile_device

    run_device = args.run_device

    step_delta = args.step_delta

    key = f"rate={rate}_cv={cv}_{image_size}"

    # no need to pass gradient
    torch.set_grad_enabled(False)
    try:
        workload_request_count = 10
        time_pattern = "Gamma"

        from datetime import datetime, timezone, timedelta
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        log_prefix = f"{log_folder}/{timestamp}_{time_pattern}_rate={rate}_cv={cv}_slo_factor={slo_factor}_request={workload_request_count}_step_delta={step_delta}_device={args.profile_device}_image_size={image_size}_"
        instant_profile_log = f"{log_folder}/{timestamp}_request={workload_request_count}_instant_step_latency.log"

        # init queue
        worker_nums = 1

        input_queue = torch.multiprocessing.Manager().Queue()
        output_queue = torch.multiprocessing.Manager().Queue()
        queue_list = [torch.multiprocessing.Manager().Queue() for _ in range(worker_nums - 1)]

        deploy_ready = torch.multiprocessing.Semaphore(0)
        
        queue_list.insert(0, input_queue)
        queue_list.append(output_queue)  

        device = run_device
        parameter_path = "./16_Network.pth"
        unet_module = UNetModule(device=device, parameter_path=parameter_path)

        # 创建工作进程
        worker_list = []
        worker_list.append(Worker(stream_module_list=[unet_module], input_queue=queue_list[0], output_queue=queue_list[1], id="DiT", log_prefix=log_prefix, deploy_ready=deploy_ready, image_size=image_size, profile_device=args.profile_device, device=device, instant_profile_log=instant_profile_log, step_slo_scale=slo_factor,  step_delta=step_delta))
    
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
        
        Palette_profile_latency = {
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

        warm_up_request = {
            "image_size": 256,
            "num-sampling-steps": 100,
            "remain_loop_count_no_cuda": 100,
            "cfg_scale": 4.0,
            "uuid": uuid.uuid1(),
            "request_time": time.time(),
            "SLO": 100,#slo_factor * Palette_profile_latency["1"] * 100,
            "id": -1 # for debug
            }
        # warm up all the unet batch_size
        test_count = 20
        for idx in range(test_count):
            #warm_up_request["id"] = idx
            time.sleep(0.1)
            temp_request = copy.deepcopy(warm_up_request)
            temp_request["request_time"] = time.time()
            temp_request["id"] = -1 - idx
            input_queue.put(temp_request)
        # warm up all the unet batch_size

        time.sleep(20)
        print("warm up succeed!")
        
        num_sampling_steps_list = gamma_process_trace_rtx4090_256["random_num_sampling_steps"]

        for idx in range(workload_request_count):
            time.sleep(arrival_interval_list[idx])
            input = {
                "image_size": 256,
                "image_ndarray": torch.rand((3, 256, 256)),
                "num-sampling-steps": num_sampling_steps_list[idx],
                "remain_loop_count_no_cuda": num_sampling_steps_list[idx],
                "cfg_scale": 4.0,
                "uuid": uuid.uuid1(),
                "request_time": time.time(),
                "SLO": slo_factor * Palette_profile_latency["1"] * num_sampling_steps_list[idx],
                "id": idx # for debug
            }
            input_queue.put(input)
            # time.sleep(20)
            print(f"input_queue put request id: {idx}""\n""-----------------------")
        # end request
        time.sleep(20) # wait for all done
        input_queue.put(None)
        
        for _worker in worker_list:
            _worker.join()

    except KeyboardInterrupt:
        print("-"*10, "Main process received KeyboardInterrupt","-"*10)