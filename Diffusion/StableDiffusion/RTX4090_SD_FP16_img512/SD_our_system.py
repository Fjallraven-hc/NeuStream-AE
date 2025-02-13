import uuid
import time
import torch
import json
from stable_diffusion_v1_5.stable_diffusion_pipeline import StableDiffusionPipeline
from stable_diffusion_v1_5.stable_diffusion_scheduler import StableDiffusionScheduler
from utils import *

def handle_output(output_queue, log_name, workload_request_count, rate, cv, slo_factor, trace_time):

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
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            statistics = f"time:{formatted_time}, Stable Diffusion 256x image, rate:{rate} qps, cv={cv}, slo={slo_factor}, NeuStream goodput_rate={goodput_request_count}/{workload_request_count}, goodput speed={goodput_request_count / trace_time}\n"
            print(statistics)
            result_file = open("stable_diffusion_serve_result.txt", "a")
            result_file.write(statistics)
            break
        request_count += 1
        print(f"Collector handle request_count: {request_count}")
        # warm-up request
        if result["id"] < 0:
            continue

        goodput_request_count += 1
        #print("-"*20)
        f.write("-"*20+"\n")
        result["finish_time"] = time.time()
        f.write(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}\n")
        print(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}")
        f.write(f'request step: {result["loop_num"]["UNetModule"]}')
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
    parser.add_argument('--slo_scale', required=True, type=str, help='a value to determine slo factor')
    parser.add_argument('--extra_vae_safety_time', required=True, type=float, help='a value to determine extra budget for vae and safety')
    parser.add_argument('--log_folder', required=True, type=str, help='a value to determine log folder')
    parser.add_argument('--profile_device', required=True, type=str, help='a value to determine profile device')
    parser.add_argument('--step_delta', required=True, type=float, help='a value to determine running device')

    args = parser.parse_args()

    key = f"rate={args.rate_scale}_cv={args.cv_scale}_{args.image_size}"

    rate = float(args.rate_scale)

    slo_factor = float(args.slo_scale)

    cv = float(args.cv_scale)

    image_size = args.image_size
    
    extra_vae_safety_time = args.extra_vae_safety_time

    log_folder = args.log_folder

    step_delta = args.step_delta

    module_latency_h100_256 = {"clip": {"1": 0.010763356614173675, "2": 0.011122808831327774, "3": 0.011640708931550687, "4": 0.011705784495764723, "5": 0.012307808962555565, "6": 0.01250345816797748, "7": 0.012666157484814828, "8": 0.012823822300719179, "9": 0.0126037277205258, "10": 0.012753783903540448, "11": 0.012932783448474204, "12": 0.013128956403345503, "13": 0.013358378706842053, "14": 0.012949418878027549, "15": 0.013127381081118229}, "unet": {"1": 0.0254171684156267, "2": 0.025889276108723516, "3": 0.026742454751261642, "4": 0.028399059777053034, "5": 0.02930485231003591, "6": 0.030402309508347998, "7": 0.031128080295664922, "8": 0.03151123231390909, "9": 0.032681762183807334, "10": 0.03391041530638325, "11": 0.03527436563174943, "12": 0.03681413127983711, "13": 0.03926054316059667, "14": 0.04135559788163828, "15": 0.04361406611088587}, "vae": {"1": 0.008543856308928557, "2": 0.013037726454132674, "3": 0.016666030511260033, "4": 0.01988933689664213, "5": 0.023583694494196346, "6": 0.026658064647748763, "7": 0.0303262718097896, "8": 0.03371112884915605, "9": 0.03733311519406888, "10": 0.040571440087289225, "11": 0.04389081437292756, "12": 0.047191714449804655, "13": 0.051138482797814876, "14": 0.0542412581378404, "15": 0.05792580714107168}, "safety": {"1": 0.027158761435017293, "2": 0.04097433093631146, "3": 0.053615434657937534, "4": 0.06755093234824017, "5": 0.08050565083976835, "6": 0.0936376547098768, "7": 0.10706905651992808, "8": 0.12038751328958476, "9": 0.13502245724278814, "10": 0.14927166088068106, "11": 0.1636664044468644, "12": 0.17882538171640286, "13": 0.1926385347751227, "14": 0.2070075457400464, "15": 0.2225177216529846}}
    module_latency_h100_512 = {"clip": {"1": 0.010729043155300374, "2": 0.011114207970700701, "3": 0.01162113156169653, "4": 0.011685678735375404, "5": 0.01228585111675784, "6": 0.012452897919836094, "7": 0.012583264557179064, "8": 0.012737178839743138, "9": 0.012580370473466357, "10": 0.012718318921929978, "11": 0.012892252879635411, "12": 0.013105942669542545, "13": 0.013317528960047936, "14": 0.012893355878380438, "15": 0.013091435142773755}, "unet": {"1": 0.02640514894939807, "2": 0.03186587899999351, "3": 0.04127401993514932, "4": 0.04956639879288114, "5": 0.06105041731985248, "6": 0.06856303299233939, "7": 0.07873859376801799, "8": 0.08774662281696995, "9": 0.09964735091974337, "10": 0.10881113736346985, "11": 0.11614277953049168, "12": 0.12608531462804726, "13": 0.13460653038540235, "14": 0.14421168505214155, "15": 0.1532588206270399}, "vae": {"1": 0.025981670264534806, "2": 0.04397096617945603, "3": 0.05861139535067641, "4": 0.07297860169593169, "5": 0.08819514516817063, "6": 0.1027777564654849, "7": 0.11762863239843627, "8": 0.13209553646417904, "9": 0.14748582096459964, "10": 0.16170084267398532, "11": 0.17982623330792602, "12": 0.19494961027284058, "13": 0.21056599848504579, "14": 0.2290685014070376, "15": 0.24449221099897914}, "safety": {"1": 0.03995978668788258, "2": 0.06576067601175357, "3": 0.09061296585392445, "4": 0.11743271744732435, "5": 0.14353326063755215, "6": 0.16942864021628487, "7": 0.19585907678507888, "8": 0.2224162568260605, "9": 0.2503597050284346, "10": 0.3000266294173421, "11": 0.3281733137167369, "12": 0.359497005022907, "13": 0.39078635129393363, "14": 0.41930771552558455, "15": 0.45168329624736564}}
    
    module_latency_rtx4090_256 = {"clip": {"1": 0.0118803435898557, "2": 0.011962506376827756, "3": 0.01214819938953345, "4": 0.012388100595368693, "5": 0.012434979273500492, "6": 0.012552506018740436, "7": 0.01274759128214197, "8": 0.01278388589544564, "9": 0.012960291467607021, "10": 0.013081496891876062, "11": 0.013141622462747048, "12": 0.013248598106050244, "13": 0.013392995586808848, "14": 0.013566676071103739, "15": 0.013798098739547035}, "unet": {"1": 0.026020232245934253, "2": 0.026309458348824054, "3": 0.027279645413616483, "4": 0.028334970471962373, "5": 0.029459379917504837, "6": 0.03157772256859711, "7": 0.03708117858183627, "8": 0.0387461300825282, "9": 0.04283448979638669, "10": 0.04627723328541128, "11": 0.04998181570245295, "12": 0.05545559861906329, "13": 0.064466998175097, "14": 0.06930127025892337, "15": 0.07160989350328843}, "vae": {"1": 0.010221718012222223, "2": 0.01745897560019274, "3": 0.02668315848829795, "4": 0.03403807218585696, "5": 0.04318167651262211, "6": 0.04924627051365619, "7": 0.05791144092016074, "8": 0.06594540321446479, "9": 0.07520562678109854, "10": 0.08338227942205813, "11": 0.09421624241358771, "12": 0.10264805575110475, "13": 0.11346113666587947, "14": 0.12186390610069645, "15": 0.13002186361700296}, "safety": {"1": 0.0212851593993148, "2": 0.02896239591420305, "3": 0.03645762749852575, "4": 0.04431993306692069, "5": 0.053565024175559695, "6": 0.0645589823058496, "7": 0.07236240844109229, "8": 0.08188239360849063, "9": 0.09085536837253881, "10": 0.10069203665670083, "11": 0.11109578200072671, "12": 0.1210063308743494, "13": 0.1295798385205368, "14": 0.13900123598674932, "15": 0.14750750624807552}}
    module_latency_rtx4090_512 = {"clip": {"1": 0.011820603545811308, "2": 0.011914930315735053, "3": 0.012091079537714194, "4": 0.01225634121495967, "5": 0.012364806635730762, "6": 0.012438555403302113, "7": 0.012616275870852699, "8": 0.012669404601734695, "9": 0.012887208054613585, "10": 0.013073412005347436, "11": 0.013164699712598866, "12": 0.013328941836764063, "13": 0.01349070567600242, "14": 0.01365713698018079, "15": 0.0138201993711368}, "unet": {"1": 0.027626497071966453, "2": 0.04298253385632327, "3": 0.055487241685352844, "4": 0.07376617182874018, "5": 0.0899044096215882, "6": 0.10230828558725089, "7": 0.12511090516836057, "8": 0.14676035077997832, "9": 0.15950222194401753, "10": 0.1782825976713664, "11": 0.19350428828931968, "12": 0.22182900357445834, "13": 0.24550344166581076, "14": 0.2606973672859521, "15": 0.27655148591300605}, "vae": {"1": 0.03376312875399488, "2": 0.07399217916849121, "3": 0.11004193404680071, "4": 0.14354593551192943, "5": 0.17946397196879696, "6": 0.21222230604835574, "7": 0.24667370498105132, "8": 0.2812068222449894, "9": 0.31662720887997775, "10": 0.3520306087460141, "11": 0.39040729648728467, "12": 0.4260003667201866, "13": 0.47153294856477646, "14": 0.5068235329185159, "15": 0.5415168143338477}, "safety": {"1": 0.03063526479106153, "2": 0.04797388670138187, "3": 0.0672388456319694, "4": 0.08437921952686667, "5": 0.1039594023502502, "6": 0.12115485408823147, "7": 0.1391513596042759, "8": 0.16103985712797653, "9": 0.17952639864861664, "10": 0.22299047340615297, "11": 0.2438703968977699, "12": 0.27341513347404833, "13": 0.2970558778869773, "14": 0.31895873308754885, "15": 0.3440723231580761}}

    trace = json.load(open("RTX4090_SD_FP16_img512_trace.json"))
    arrival_interval_list = trace[f"rate={args.rate_scale},cv={args.cv_scale}"]
    random_step_list = trace["random_step_list"]

    if args.profile_device == "h100":
        if image_size == 256:
            module_latency_variable = module_latency_h100_256
            # gamma_process_interval_variable = gamma_process_trace_h100_256[key]
        elif image_size == 512:
            module_latency_variable = module_latency_h100_512
            # gamma_process_interval_variable = gamma_process_trace_h100_512[key]
    elif args.profile_device == "rtx4090":
        if image_size == 256:
            module_latency_variable = module_latency_rtx4090_256
            # gamma_process_interval_variable = gamma_process_trace_rtx4090_256[key]
        if image_size == 512:
            module_latency_variable = module_latency_rtx4090_512
            # gamma_process_interval_variable = gamma_process_trace_rtx4090_512[key]

    # no need to pass gradient
    torch.set_grad_enabled(False)
    try:
        # init pipeline from config
        sd_config_file = "stable_diffusion_v1_5/config.json"
        sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

        # init scheduler
        sd_scheduler = StableDiffusionScheduler()

        # trace setting
        from test_set import prompt_list

        #time_pattern = "uniform"
        #time_pattern = "new_gamma_request=10"
        time_pattern = "request=500"
        workload_request_count = 500

        #delay_num = args.delay_num
        from datetime import datetime, timezone, timedelta
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        log_prefix = f"{log_folder}/{timestamp}_image_size={image_size}_{time_pattern}_rate={rate}_cv={cv}_slo_factor={slo_factor}_extra_vae_time={extra_vae_safety_time}_device={args.profile_device}_step_delta={step_delta}"

        # init queue
        worker_nums = 3
    
        input_queue = torch.multiprocessing.Manager().Queue()
        output_queue = torch.multiprocessing.Manager().Queue()
        queue_list = [torch.multiprocessing.Manager().Queue() for _ in range(worker_nums - 1)]

        deploy_ready = torch.multiprocessing.Semaphore(0)
        
        #input_queue = queue.Queue()
        #output_queue = queue.Queue()
        #queue_list = [queue.Queue() for _ in range(worker_nums - 1)]
        queue_list.insert(0, input_queue)
        queue_list.append(output_queue)        

        device = "cuda:1"
        # 创建工作进程
        worker_list = []
        worker_list.append(Worker(stream_module_list=sd_pipeline.stream_module_list[0:1], input_queue=queue_list[0], output_queue=queue_list[1], id="clip", log_prefix=log_prefix, deploy_ready=deploy_ready, extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, step_slo_scale=slo_factor,  step_delta=step_delta))
        worker_list.append(Worker(stream_module_list=sd_pipeline.stream_module_list[1:2], input_queue=queue_list[1], output_queue=queue_list[2], id="unet", log_prefix=log_prefix, deploy_ready=deploy_ready, extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, step_slo_scale=slo_factor,  step_delta=step_delta))
        worker_list.append(Worker(stream_module_list=sd_pipeline.stream_module_list[2:], input_queue=queue_list[2], output_queue=queue_list[3], id="vae&safety", log_prefix=log_prefix, deploy_ready=deploy_ready, extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, step_slo_scale=slo_factor,  step_delta=step_delta))

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
        
        warm_up_request = {
            "prompt": "a beautiful girl studying in Chinese University",
            "height": image_size,
            "width": image_size,
            "loop_num": {
                "UNetModule": 50
            },
            "request_time": time.time(),
            "guidance_scale": 7.5,
            "seed": 0,
            "SLO": 10000,
            "loop_index": {
                "UNetModule": 0
            },
            "id": -1
        }
        # warm up all the unet batch_size
        test_count = 15
        for idx in range(test_count):
            #warm_up_request["id"] = idx
            time.sleep(0.3)
            warm_up_request["request_time"] = time.time()
            input_queue.put(warm_up_request)
        # warm up all the unet batch_size
        """
        # warm up all the unet batch_size
        # threading version, because in-process queue only pass reference
        for _ in range(10):
            input_queue.put(copy.deepcopy(warm_up_request))
            time.sleep(0.8)"""
        
        # sleep until warm-up end
        time.sleep(20)
        print("warm up succeed!")

        #steps_range = list(range(-5, 6))

        for idx in range(workload_request_count):
            time.sleep(arrival_interval_list[idx])
            input = {
                "prompt": prompt_list[idx%100],
                "height": image_size,
                "width": image_size,
                "loop_num": {
                    "UNetModule": random_step_list[idx],#50 + steps_range[idx % 11]#steps_list[idx%100]
                },
                "loop_index": {
                    "UNetModule": 0
                },
                "guidance_scale": 7.5,
                "seed": 0,
                "uuid": uuid.uuid1(),
                "request_time": time.time(),
                "SLO": slo_factor * (module_latency_variable["clip"]["1"] + module_latency_variable["unet"]["1"] * random_step_list[idx] + module_latency_variable["vae"]["1"] + module_latency_variable["safety"]["1"]),
                "id": idx # for debug
            }
            input_queue.put(input)
            print(f"clip_queue put item: {input['id']}\n-----------------------")
        # end request
        time.sleep(10)
        input_queue.put(None)
        
        for _worker in worker_list:
            _worker.join()

    except KeyboardInterrupt:
        print("-"*10, "Main process received KeyboardInterrupt","-"*10)