## ClockWork 
import uuid
from dataclasses import dataclass
from tqdm.auto import tqdm
import torch
import time
import multiprocessing
from diffusers import StableDiffusionPipeline,EulerAncestralDiscreteScheduler
import threading


import argparse

parser = argparse.ArgumentParser(description="parameter for server.")

parser.add_argument('--image_size', required=True, type=int, help='a value to determine image size')
parser.add_argument('--rate_scale', required=True, type=str, help='a value to determine arrival rate')
parser.add_argument('--cv_scale', required=True, type=str, help='a value to determine arrival coefficient of variation')
parser.add_argument('--slo_scale', required=True, type=str, help='a value to determine slo factor')
parser.add_argument('--log_folder', required=True, type=str, help='a value to determine log folder')
parser.add_argument('--device', required=True, type=str, help='a value to determine running device')

args = parser.parse_args()

# profile latency can be simulated by module latency
module_latency_h100_256 = {"clip": {"1": 0.010763356614173675, "2": 0.011122808831327774, "3": 0.011640708931550687, "4": 0.011705784495764723, "5": 0.012307808962555565, "6": 0.01250345816797748, "7": 0.012666157484814828, "8": 0.012823822300719179, "9": 0.0126037277205258, "10": 0.012753783903540448, "11": 0.012932783448474204, "12": 0.013128956403345503, "13": 0.013358378706842053, "14": 0.012949418878027549, "15": 0.013127381081118229}, "unet": {"1": 0.0254171684156267, "2": 0.025889276108723516, "3": 0.026742454751261642, "4": 0.028399059777053034, "5": 0.02930485231003591, "6": 0.030402309508347998, "7": 0.031128080295664922, "8": 0.03151123231390909, "9": 0.032681762183807334, "10": 0.03391041530638325, "11": 0.03527436563174943, "12": 0.03681413127983711, "13": 0.03926054316059667, "14": 0.04135559788163828, "15": 0.04361406611088587}, "vae": {"1": 0.008543856308928557, "2": 0.013037726454132674, "3": 0.016666030511260033, "4": 0.01988933689664213, "5": 0.023583694494196346, "6": 0.026658064647748763, "7": 0.0303262718097896, "8": 0.03371112884915605, "9": 0.03733311519406888, "10": 0.040571440087289225, "11": 0.04389081437292756, "12": 0.047191714449804655, "13": 0.051138482797814876, "14": 0.0542412581378404, "15": 0.05792580714107168}, "safety": {"1": 0.027158761435017293, "2": 0.04097433093631146, "3": 0.053615434657937534, "4": 0.06755093234824017, "5": 0.08050565083976835, "6": 0.0936376547098768, "7": 0.10706905651992808, "8": 0.12038751328958476, "9": 0.13502245724278814, "10": 0.14927166088068106, "11": 0.1636664044468644, "12": 0.17882538171640286, "13": 0.1926385347751227, "14": 0.2070075457400464, "15": 0.2225177216529846}}
module_latency_h100_512 = {"clip": {"1": 0.010729043155300374, "2": 0.011114207970700701, "3": 0.01162113156169653, "4": 0.011685678735375404, "5": 0.01228585111675784, "6": 0.012452897919836094, "7": 0.012583264557179064, "8": 0.012737178839743138, "9": 0.012580370473466357, "10": 0.012718318921929978, "11": 0.012892252879635411, "12": 0.013105942669542545, "13": 0.013317528960047936, "14": 0.012893355878380438, "15": 0.013091435142773755}, "unet": {"1": 0.02640514894939807, "2": 0.03186587899999351, "3": 0.04127401993514932, "4": 0.04956639879288114, "5": 0.06105041731985248, "6": 0.06856303299233939, "7": 0.07873859376801799, "8": 0.08774662281696995, "9": 0.09964735091974337, "10": 0.10881113736346985, "11": 0.11614277953049168, "12": 0.12608531462804726, "13": 0.13460653038540235, "14": 0.14421168505214155, "15": 0.1532588206270399}, "vae": {"1": 0.025981670264534806, "2": 0.04397096617945603, "3": 0.05861139535067641, "4": 0.07297860169593169, "5": 0.08819514516817063, "6": 0.1027777564654849, "7": 0.11762863239843627, "8": 0.13209553646417904, "9": 0.14748582096459964, "10": 0.16170084267398532, "11": 0.17982623330792602, "12": 0.19494961027284058, "13": 0.21056599848504579, "14": 0.2290685014070376, "15": 0.24449221099897914}, "safety": {"1": 0.03995978668788258, "2": 0.06576067601175357, "3": 0.09061296585392445, "4": 0.11743271744732435, "5": 0.14353326063755215, "6": 0.16942864021628487, "7": 0.19585907678507888, "8": 0.2224162568260605, "9": 0.2503597050284346, "10": 0.3000266294173421, "11": 0.3281733137167369, "12": 0.359497005022907, "13": 0.39078635129393363, "14": 0.41930771552558455, "15": 0.45168329624736564}}

module_latency_rtx4090_256 = {"clip": {"1": 0.0118803435898557, "2": 0.011962506376827756, "3": 0.01214819938953345, "4": 0.012388100595368693, "5": 0.012434979273500492, "6": 0.012552506018740436, "7": 0.01274759128214197, "8": 0.01278388589544564, "9": 0.012960291467607021, "10": 0.013081496891876062, "11": 0.013141622462747048, "12": 0.013248598106050244, "13": 0.013392995586808848, "14": 0.013566676071103739, "15": 0.013798098739547035}, "unet": {"1": 0.026020232245934253, "2": 0.026309458348824054, "3": 0.027279645413616483, "4": 0.028334970471962373, "5": 0.029459379917504837, "6": 0.03157772256859711, "7": 0.03708117858183627, "8": 0.0387461300825282, "9": 0.04283448979638669, "10": 0.04627723328541128, "11": 0.04998181570245295, "12": 0.05545559861906329, "13": 0.064466998175097, "14": 0.06930127025892337, "15": 0.07160989350328843}, "vae": {"1": 0.010221718012222223, "2": 0.01745897560019274, "3": 0.02668315848829795, "4": 0.03403807218585696, "5": 0.04318167651262211, "6": 0.04924627051365619, "7": 0.05791144092016074, "8": 0.06594540321446479, "9": 0.07520562678109854, "10": 0.08338227942205813, "11": 0.09421624241358771, "12": 0.10264805575110475, "13": 0.11346113666587947, "14": 0.12186390610069645, "15": 0.13002186361700296}, "safety": {"1": 0.0212851593993148, "2": 0.02896239591420305, "3": 0.03645762749852575, "4": 0.04431993306692069, "5": 0.053565024175559695, "6": 0.0645589823058496, "7": 0.07236240844109229, "8": 0.08188239360849063, "9": 0.09085536837253881, "10": 0.10069203665670083, "11": 0.11109578200072671, "12": 0.1210063308743494, "13": 0.1295798385205368, "14": 0.13900123598674932, "15": 0.14750750624807552}}
# below is partly modified by instant profiled data
module_latency_rtx4090_256 = {"clip": {"1": 0.0118803435898557, "2": 0.011962506376827756, "3": 0.01214819938953345, "4": 0.012388100595368693, "5": 0.012434979273500492, "6": 0.012552506018740436, "7": 0.01274759128214197, "8": 0.01278388589544564, "9": 0.012960291467607021, "10": 0.013081496891876062, "11": 0.013141622462747048, "12": 0.013248598106050244, "13": 0.013392995586808848, "14": 0.013566676071103739, "15": 0.013798098739547035}, "unet": {"1": 0.03305803346058762, "2": 0.03246075380717308, "3": 0.03334193337962546, "4": 0.03592328585807958, "5": 0.034633072449123815, "6": 0.0367765142131717, "7": 0.03872685558786711, "8": 0.03945022503580524, "9": 0.04687323158523485, "10": 0.04627723328541128, "11": 0.04998181570245295, "12": 0.05545559861906329, "13": 0.064466998175097, "14": 0.06930127025892337, "15": 0.07160989350328843}, "vae": {"1": 0.010221718012222223, "2": 0.01745897560019274, "3": 0.02668315848829795, "4": 0.03403807218585696, "5": 0.04318167651262211, "6": 0.04924627051365619, "7": 0.05791144092016074, "8": 0.06594540321446479, "9": 0.07520562678109854, "10": 0.08338227942205813, "11": 0.09421624241358771, "12": 0.10264805575110475, "13": 0.11346113666587947, "14": 0.12186390610069645, "15": 0.13002186361700296}, "safety": {"1": 0.0212851593993148, "2": 0.02896239591420305, "3": 0.03645762749852575, "4": 0.04431993306692069, "5": 0.053565024175559695, "6": 0.0645589823058496, "7": 0.07236240844109229, "8": 0.08188239360849063, "9": 0.09085536837253881, "10": 0.10069203665670083, "11": 0.11109578200072671, "12": 0.1210063308743494, "13": 0.1295798385205368, "14": 0.13900123598674932, "15": 0.14750750624807552}}


module_latency_rtx4090_512 = {"clip": {"1": 0.011820603545811308, "2": 0.011914930315735053, "3": 0.012091079537714194, "4": 0.01225634121495967, "5": 0.012364806635730762, "6": 0.012438555403302113, "7": 0.012616275870852699, "8": 0.012669404601734695, "9": 0.012887208054613585, "10": 0.013073412005347436, "11": 0.013164699712598866, "12": 0.013328941836764063, "13": 0.01349070567600242, "14": 0.01365713698018079, "15": 0.0138201993711368}, "unet": {"1": 0.027626497071966453, "2": 0.04298253385632327, "3": 0.055487241685352844, "4": 0.07376617182874018, "5": 0.0899044096215882, "6": 0.10230828558725089, "7": 0.12511090516836057, "8": 0.14676035077997832, "9": 0.15950222194401753, "10": 0.1782825976713664, "11": 0.19350428828931968, "12": 0.22182900357445834, "13": 0.24550344166581076, "14": 0.2606973672859521, "15": 0.27655148591300605}, "vae": {"1": 0.03376312875399488, "2": 0.07399217916849121, "3": 0.11004193404680071, "4": 0.14354593551192943, "5": 0.17946397196879696, "6": 0.21222230604835574, "7": 0.24667370498105132, "8": 0.2812068222449894, "9": 0.31662720887997775, "10": 0.3520306087460141, "11": 0.39040729648728467, "12": 0.4260003667201866, "13": 0.47153294856477646, "14": 0.5068235329185159, "15": 0.5415168143338477}, "safety": {"1": 0.03063526479106153, "2": 0.04797388670138187, "3": 0.0672388456319694, "4": 0.08437921952686667, "5": 0.1039594023502502, "6": 0.12115485408823147, "7": 0.1391513596042759, "8": 0.16103985712797653, "9": 0.17952639864861664, "10": 0.22299047340615297, "11": 0.2438703968977699, "12": 0.27341513347404833, "13": 0.2970558778869773, "14": 0.31895873308754885, "15": 0.3440723231580761}}
# below is instant profiled data
module_latency_rtx4090_512 = {"clip": {"1": 0.011820603545811308, "2": 0.011914930315735053, "3": 0.012091079537714194, "4": 0.01225634121495967, "5": 0.012364806635730762, "6": 0.012438555403302113, "7": 0.012616275870852699, "8": 0.012669404601734695, "9": 0.012887208054613585, "10": 0.013073412005347436, "11": 0.013164699712598866, "12": 0.013328941836764063, "13": 0.01349070567600242, "14": 0.01365713698018079, "15": 0.0138201993711368}, "unet": {"1": 0.029338991034390603, "2": 0.04098136983737797, "3": 0.053191183675822096, "4": 0.06900042770920613, "5": 0.08510530272953327, "6": 0.09226540044288743, "7": 0.11469343446647923, "8": 0.13317389850677966, "9": 0.14125110079062939, "10": 0.154387593461091, "11": 0.16638505648307758, "12": 0.19492023764178154, "13": 0.21799246336405093, "14": 0.2606973672859521, "15": 0.27655148591300605}, "vae": {"1": 0.03376312875399488, "2": 0.07399217916849121, "3": 0.11004193404680071, "4": 0.14354593551192943, "5": 0.17946397196879696, "6": 0.21222230604835574, "7": 0.24667370498105132, "8": 0.2812068222449894, "9": 0.31662720887997775, "10": 0.3520306087460141, "11": 0.39040729648728467, "12": 0.4260003667201866, "13": 0.47153294856477646, "14": 0.5068235329185159, "15": 0.5415168143338477}, "safety": {"1": 0.03063526479106153, "2": 0.04797388670138187, "3": 0.0672388456319694, "4": 0.08437921952686667, "5": 0.1039594023502502, "6": 0.12115485408823147, "7": 0.1391513596042759, "8": 0.16103985712797653, "9": 0.17952639864861664, "10": 0.22299047340615297, "11": 0.2438703968977699, "12": 0.27341513347404833, "13": 0.2970558778869773, "14": 0.31895873308754885, "15": 0.3440723231580761}}

pro = {}
if args.image_size == 256:
    for steps in range(30, 51):
        pro[steps] = {}
        for batch_size in range(1, 16):
            pro[steps][batch_size] = module_latency_h100_256["clip"][str(batch_size)] + module_latency_rtx4090_256["unet"][str(batch_size)] * steps + module_latency_rtx4090_256["vae"][str(batch_size)] + module_latency_rtx4090_256["safety"][str(batch_size)]
elif args.image_size == 512:
    for steps in range(30, 51):
        pro[steps] = {}
        for batch_size in range(1, 16):
            pro[steps][batch_size] = module_latency_h100_512["clip"][str(batch_size)] + module_latency_rtx4090_512["unet"][str(batch_size)] * steps + module_latency_rtx4090_512["vae"][str(batch_size)] + module_latency_rtx4090_512["safety"][str(batch_size)]

@dataclass
class Strategy:
    steps: int
    # img_size:int
    batch_size: int
    latest: float

## 需要有一个controller,进行调度，这里我先假设模型都在显存中
def controller(request_queue,infer_queue,num_worker, msg_queue):
    sd = {
            30:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            31:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            32:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            33:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            34:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            35:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            36:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            37:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            38:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            39:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            40:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            41:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            42:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            43:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            44:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            45:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            46:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            47:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            48:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            49:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
            50:{
                1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]
            },
        }
    batch_size = [1,2,3,4,5,]
    # img_size = [256,512,768,1024]
    outstanding_work = [0 for _ in range(num_worker)]
    #strategy_list = [[]]
    strategy_list = [[] for _ in range(num_worker)]
    # flag = [False,False,False,False,False]
    flag = [False for _ in range(num_worker)]
    lock = threading.Lock()
    lock2 = threading.Lock()
    import queue
    thread_msg_queue = queue.Queue()
    t1 = threading.Thread(target=check_outstanding_work,args=(outstanding_work,flag,lock, thread_msg_queue))
    t2 = threading.Thread(target=check_batch,args=(sd,lock2, thread_msg_queue))
    t1.start()
    t2.start()
    while True: 
        if not msg_queue.empty():
            thread_msg_queue.put({"msg": "end serving"})
            time.sleep(1)
            break
        for (idx,f) in enumerate(flag):
            if f:
                for steps in range(30,51):
                    # for size in img_size:
                    for b in batch_size:
                        if len(sd[steps][b]) == 0:
                            continue

                        ddl = sd[steps][b][0]["request_time"] + sd[steps][b][0]["SLOs"]

                        strategy_list[idx].append(Strategy(steps=steps,batch_size=b,latest=ddl-pro[steps][b]))            
            if len(strategy_list[idx]) > 0:
                strategy_list[idx].sort(key=lambda x:x.latest)
                for (id,strategy) in enumerate(strategy_list[idx]):
                    if time.time() <= strategy.latest and len(sd[strategy.steps][strategy.batch_size]) == strategy.batch_size:
                        input = []
                        for i in range(strategy.batch_size):
                            ## 添加input，同时去除sd中的item
                            item = sd[strategy.steps][strategy.batch_size].pop(0)
                            input.append(item)
                            ##
                            id_item = item["id"]
                            for b in batch_size:
                                if b == strategy.batch_size:
                                    continue
                                with lock2:
                                    sd[strategy.steps][b] = list(filter(lambda x:x["id"] != id_item,sd[strategy.steps][b]))
                        outstanding_work[idx] = time.time() + pro[strategy.steps][strategy.batch_size]
                        infer_queue[idx].put(input) 
                        with lock: 
                            flag[idx] = False
                        strategy_list[idx] = []
                        break
                        
        if request_queue.empty():
            continue
        else:
            item = request_queue.get()
            for b in batch_size:
                # if item["height"] == 1024 and b == 2:
                #     continue
                with lock2:
                    sd[item["inference_steps"]][b].append(item)  

def check_outstanding_work(outstanding_work,flag,lock,msg_queue):
    while True:
        for (id,ow) in enumerate(outstanding_work):
            if ow - time.time() <= 0.005:
                ## 需要调度
                with lock:
                    flag[id] = True
                # print(f"{id}号worker需要调度")
        if not msg_queue.empty():
            break
        time.sleep(0.001)

def check_batch(sd_dict,lock,msg_queue):
    while True:
        for steps in sd_dict.keys():
            for batch in sd_dict[steps].keys():
                with lock:
                    for item in sd_dict[steps][batch]:
                        if item["request_time"] + item["SLOs"] < time.time()+pro[steps][batch]:
                            #with lock:
                            sd_dict[steps][batch].remove(item)
                            ## 这里需要发出一个action，告诉模型，这个infer已经被丢弃了                      
        if not msg_queue.empty():
            break
        time.sleep(0.003)

@torch.no_grad()
def infer_worker(infer_queue,pipe,idxx, log_prefix, msg_queue):
    #benchmark_log = open(f"clockwork_worker_{idxx}_microbenchmark_gamma_{log_prefix}.log", "w")
    benchmark_log = open(f"{log_prefix}_clockwork_worker_{idxx}.log", "w")
    good = 0
    finish = 0
    abandon = 0
    ## warm up
    # model_path =  "/home/yhc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0"
    # pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:0")
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe("girl",num_inference_steps=30)
    pipe("girl",num_inference_steps=35)
    import time           
    while True:
        if not msg_queue.empty():
            break
        if infer_queue.empty():
            continue
        item = infer_queue.get()
        #if "msg" in item:
        #    break
        batch_size = len(item)
        # ddl = item[0]["request_time"] + item[0]["SLOs"]
        steps = item[0]["inference_steps"]
        prompt = []
        expected = pro[steps][batch_size] 
        for i in item:
            prompt.append(i["prompt"])
        h = item[0]["height"]
        w=h
        st = time.time()
        pipe(prompt,num_inference_steps=steps,height=h,width=w)
        ed = time.time()
        for i in item:
            benchmark_log.write(f'request: {i["yhc_id"]}, latency: {ed - i["request_time"]}\n')
        finish += batch_size
        for i in item:
            if ed <= i["request_time"] + i["SLOs"]:
                good += 1
        # 记录batch的request id
        
        sid = ""
        for i in item:
            sid += str(i["id"]) + " "
        benchmark_log.write(f"{idxx}号worker ---- good : {good} --- finish: {finish} ---- abandon : {abandon} ---- batch size : {len(item)}  --- ed : {ed} --- --- id: {sid} actual:{ed-st}  expected:{expected}\n")
        benchmark_log.flush()

if __name__ == "__main__":
    model_path = "../model_parameters/hf_format"

    import json

    RTX4090_SD_FP16_img512_trace = json.load(open("H100_SD_FP16_img512_trace.json"))

    rate = float(args.rate_scale)

    slo_factor = float(args.slo_scale)

    image_size = args.image_size   

    cv = float(args.cv_scale)

    log_folder = args.log_folder

    key = f"rate={rate}_cv={cv}_{image_size}"

    arrival_interval_list = RTX4090_SD_FP16_img512_trace[f"rate={args.rate_scale},cv={args.cv_scale}"]
    random_step_list = RTX4090_SD_FP16_img512_trace["random_step_list"]
    
    if image_size == 256:
        module_latency_variable = module_latency_h100_256
    if image_size == 512:
        module_latency_variable = module_latency_h100_512
    
    loop_count = 500

    from datetime import datetime, timezone, timedelta
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    #log_prefix = f"new_gamma_image_size={image_size}_request={loop_count}_cv={cv}_slo_factor={slo_factor}_{timestamp}"
    log_prefix = f"{log_folder}/gamma_process_image_size={image_size}_request={loop_count}_rate={rate}_cv={cv}_slo_factor={slo_factor}_{timestamp}"

    torch.multiprocessing.set_start_method("spawn")
    num_worker = 1
    pipes = []
    device = args.device
    for idx in range(num_worker):
        pipes.append(StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device))
    #     pipes = [        
    #     StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:2"),
    #     StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:1"),
    #     StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:0"),
    #     StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:7"),
    # ]
    for pipe in pipes:        
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    infer_queue = [multiprocessing.Queue() for _ in range(num_worker)]
    request_queue = multiprocessing.Queue()
    msg_queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(target=infer_worker, args=(infer_queue[0],pipes[0],0,log_prefix, msg_queue)),
        #multiprocessing.Process(target=infer_worker, args=(infer_queue[1],pipes[1],1)),
        #multiprocessing.Process(target=infer_worker, args=(infer_queue[2],pipes[2],2)),
        #multiprocessing.Process(target=infer_worker, args=(infer_queue[3],pipes[3],3)),
        # multiprocessing.Process(target=infer_worker, args=(infer_queue[4],pipes[4],4)),
        multiprocessing.Process(target=controller, args=(request_queue,infer_queue,num_worker, msg_queue)),
    ]

    for p in processes:
        p.start()
    # delay = [1,3,5,7,9]
    # delay_num = delay[0]
    # with open("log_clockwork.txt","a+") as f:
    #     f.write(f"------------delay_num:{delay_num}  num_worker:{num_worker}-------------------------\n")
    log_file = False
    # print("Successfully!")
    ## warm up
    time.sleep(20)
    print("begin infer.")

    from test_set import prompt_list

    input = {
        "prompt": prompt_list[idx%100],
        "height": image_size,
        "width": image_size,
        "inference_steps":50,
        "guidance_scale": 7.5,
        "uuid": uuid.uuid1(),
        "request_time": time.time(),
        "SLOs": slo_factor * (module_latency_variable["clip"]["1"] + module_latency_variable["unet"]["1"] * 50 + module_latency_variable["vae"]["1"] + module_latency_variable["safety"]["1"]),
        "id": 0,# for debug,
        "yhc_id": 0
    }
    
    begin = time.time()
    print(f"loop_count:{loop_count}")
    begin = time.time()
    
    for idx in range(loop_count):
        time.sleep(arrival_interval_list[idx])
        input = {
            "prompt": prompt_list[idx%100],
            "height": image_size,#512,
            "width": image_size,#512,
            "inference_steps": random_step_list[idx],
            "guidance_scale": 7.5,
            "uuid": uuid.uuid1(),
            "request_time": time.time(),
            "SLOs": slo_factor * (module_latency_variable["clip"]["1"] + module_latency_variable["unet"]["1"] * random_step_list[idx] + module_latency_variable["vae"]["1"] + module_latency_variable["safety"]["1"]),#4 * (pro[random_sample_steps[idx%loop_count]][1]-0.12),
            "id": idx ,# for debug
            "yhc_id": idx
        }
        request_queue.put(input)
    time.sleep(20)
    msg_queue.put({"msg": "end serving"})
    print(f"end infer. Begin time:{begin}")
    for p in processes:
        p.join()
    print("throughput:", loop_count / (time.time() - begin))
