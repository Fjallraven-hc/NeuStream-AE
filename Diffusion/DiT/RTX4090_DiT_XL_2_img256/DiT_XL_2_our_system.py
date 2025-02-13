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
            statistics = f"time:{timestamp}, DiT_XL_2 256x image, rate:{rate} qps, cv={cv}, slo={slo_factor}, NeuStream goodput_rate={goodput_request_count}/{workload_request_count}, goodput speed={goodput_request_count / trace_time}\n"
            print(statistics)
            result_file = open("DiT_XL_2_serve_result.txt", "a")
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

    gamma_process_trace_rtx4090_256 = json.load(open("DiT_XL_2_256_trace.json"))

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
        DiT_module = DiTModule(device=device, data_type="float16", parameter_path="../DiT_pretrained_models/DiT-XL-2-256x256.pt", DiT_config={})
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
        
        DiT_XL_2_profile_latency = {
            "DiT": {'1': 0.017604099575430156, '2': 0.017825833953917028, '3': 0.018477643452584742, '4': 0.02469173302873969, '5': 0.02942004717513919, '6': 0.03272720616683364, '7': 0.03535442834347487, '8': 0.04665777732804417, '9': 0.05022930987551808, '10': 0.051100462961941956, '11': 0.05788471630960703, '12': 0.059985206332057714, '13': 0.07121700321137905, '14': 0.06724760262295604, '15': 0.0794491237513721, '16': 0.0812934876307845, '17': 0.08649879134073854, '18': 0.09107603489607573, '19': 0.09463087095320225, '20': 0.09774496657773853},
            "vae": {"1": 0.017947486291329067, "2": 0.03042532696078221, "3": 0.04502915232442319, "4": 0.06113041636223594, "5": 0.0758535637985915, "6": 0.08969547070252398, "7": 0.10408120877109468, "8": 0.11871222325911125, "9": 0.13510771539683145, "10": 0.14904609449828665, "11": 0.16450104828303058, "12": 0.17836392776419718}
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
                                + DiT_XL_2_profile_latency["DiT"]["1"] * 250
                                + DiT_XL_2_profile_latency["vae"]["1"]
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
                                     + DiT_XL_2_profile_latency["DiT"]["1"] * num_sampling_steps_list[idx]
                                     + DiT_XL_2_profile_latency["vae"]["1"]
                                     ),
                "id": idx # for debug
            }
            input_queue.put(input)
            # time.sleep(6)
            print(f"input_queue put request id: {idx}""\n""-----------------------")
        # end request
        time.sleep(10) # wait for all done
        input_queue.put(None)
        
        for _worker in worker_list:
            _worker.join()

    except KeyboardInterrupt:
        print("-"*10, "Main process received KeyboardInterrupt","-"*10)