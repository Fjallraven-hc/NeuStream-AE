import os
import time
import queue
import torch
import torch.multiprocessing as multiprocessing
import threading
import json
from datetime import datetime, timezone, timedelta

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
    def __init__(self, stream_module_list, input_queue, output_queue, id, log_prefix, deploy_ready, extra_vae_safety_time, image_size, profile_device, step_slo_scale: float, step_delta: float, **kwargs):
        super().__init__()
        self.stream_module_list = stream_module_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.id = id # mark worker kind
        self.batch_queue = queue.PriorityQueue() # (priority, item). lowest first
        self.low_priority_batch_queue = queue.PriorityQueue()
        self.current_batch = []
        self.batch_ready = threading.Semaphore(0)
        self.batch_used = threading.Semaphore(0)
        self.first_batch = True
        self.loop_module_list = [type(stream_module).__name__ for stream_module in self.stream_module_list if stream_module.loop_module]
        module_name_list = [type(stream_module).__name__ for stream_module in self.stream_module_list]
        self.module_tag = "&".join(module_name_list)
        self.loop_unit = 1 if len(self.loop_module_list) != 0 else 1
        
        self.rtx4090_profile_latency_256 = {'clip': {'1': 0.012474376764148474, '2': 0.012474376764148474, '3': 0.012474376764148474, '4': 0.012558511504903436, '5': 0.012788628870621323, '6': 0.0129229333717376, '7': 0.012932872660458087, '8': 0.01306182999163866, '9': 0.014510652627795934, '10': 0.014318388616666199, '11': 0.015580848082900048, '12': 0.014216580241918565, '13': 0.01398760238662362, '14': 0.014092422015964985, '15': 0.014165295204147697, '16': 0.014403185434639454, '17': 0.014741145772859454, '18': 0.01453672667965293, '19': 0.01454892372712493, '20': 0.014730218267068267, '21': 0.014871926130726933, '22': 0.014901460092514753, '23': 0.01522523364983499, '24': 0.01534966727718711, '25': 0.015416669556871057, '26': 0.01574965347535908, '27': 0.015748401507735252, '28': 0.01591687535867095, '29': 0.015978783695027234, '30': 0.01725494572892785, '31': 0.016560928178951145, '32': 0.01645969459787011, '33': 0.016594909681007266, '34': 0.01677121642045677, '35': 0.017284706411883236, '36': 0.017606865325942637, '37': 0.01755879770964384, '38': 0.01776973676867783, '39': 0.018063285537064076, '40': 0.018152701165527106}, "unet": {"1": 0.03305803346058762, "2": 0.03246075380717308, "3": 0.03334193337962546, "4": 0.03592328585807958, "5": 0.034633072449123815, "6": 0.0367765142131717, "7": 0.03872685558786711, "8": 0.03945022503580524, "9": 0.04687323158523485, "10": 0.04627723328541128, "11": 0.04998181570245295, "12": 0.05545559861906329, '13': 0.06570890220813454, '14': 0.07100141234695911, '15': 0.07328080205246806, '16': 0.07591824806295336, '17': 0.08146004512906074, '18': 0.08157065802719445, '19': 0.0846785593777895, '20': 0.08814602478407324, '21': 0.09209374040365219, '22': 0.09869908122345805, '23': 0.10155065590515733, '24': 0.1039102641493082, '25': 0.11151734471321106, '26': 0.11950038983020932, '27': 0.1221927992021665, '28': 0.12548228117171675, '29': 0.12859780826140196, '30': 0.1320568386791274, '31': 0.13403273114003242, '32': 0.13587075640447438, '33': 0.14620376070495694, '34': 0.14759612683206796, '35': 0.150946809113957, '36': 0.15429785173851995, '37': 0.15909588349051773, '38': 0.161613528043963, '39': 0.1736784762982279, '40': 0.17631793598644435}, 'vae': {'1': 0.010051007685251533, '2': 0.015773355052806436, '3': 0.02298718993552029, '4': 0.031195681937970222, '5': 0.0403979274444282, '6': 0.04736588289961219, '7': 0.055613140831701456, '8': 0.06364026418887078, '9': 0.07232829523272813, '10': 0.08054304982069879, '11': 0.09089413553010672, '12': 0.09953559825662524, '13': 0.10977120802272111, '14': 0.1185326371807605, '15': 0.12669599633663892, '16': 0.1348925462923944, '17': 0.14437405702192335, '18': 0.15271617132239043, '19': 0.16071462186519056, '20': 0.1690435196971521, '21': 0.1781171625247225, '22': 0.18649539642501622, '23': 0.1897271171445027, '24': 0.20096263592131436, '25': 0.20694523139391094, '26': 0.21517597420606763, '27': 0.22299641878344117, '28': 0.23068457033950834, '29': 0.23956307345069944, '30': 0.24752458727452903, '31': 0.25550214562099427, '32': 0.26326403284911065, '33': 0.27165914939250796, '34': 0.2796974290162325, '35': 0.28824166415724906, '36': 0.30226541878655555, '37': 0.30636822546366604, '38': 0.31468988093547523, '39': 0.3229310984024778, '40': 0.33085627858527006}, 'safety': {'1': 0.0417292199190706, '2': 0.041563717965036634, '3': 0.06280139205977321, '4': 0.08411508755758405, '5': 0.10518699841573835, '6': 0.12902892417274414, '7': 0.1494296956900507, '8': 0.17310933298431336, '9': 0.19601270590908826, '10': 0.22731298906728625, '11': 0.25593423963524403, '12': 0.26365003039129076, '13': 0.28114451761357484, '14': 0.30578165215440095, '15': 0.32366123630665244, '16': 0.36839930947870014, '17': 0.3885221444722265, '18': 0.4180576408933848, '19': 0.4335001892130822, '20': 0.46123311412520707, '21': 0.5040064617991448, '22': 0.5052870756480843, '23': 0.5236162622459233, '24': 0.5464271922409535, '25': 0.5904989183787257, '26': 0.5865158761106432, '27': 0.6318265849910677, '28': 0.6753438641503453, '29': 0.708720679813996, '30': 0.7105849677789956, '31': 0.7055025860294699, '32': 0.7490916519518942, '33': 0.7440436743106693, '34': 0.7824209834448993, '35': 0.8142067297175527, '36': 0.8439213963784278, '37': 0.8747435247059911, '38': 0.9213549491483718, '39': 0.9254748675134032, '40': 0.9409719249047339}}
        self.module_latency_rtx4090_512 = {"clip": {"1": 0.011820603545811308, "2": 0.011914930315735053, "3": 0.012091079537714194, "4": 0.01225634121495967, "5": 0.012364806635730762, "6": 0.012438555403302113, "7": 0.012616275870852699, "8": 0.012669404601734695, "9": 0.012887208054613585, "10": 0.013073412005347436, "11": 0.013164699712598866, "12": 0.013328941836764063, "13": 0.01349070567600242, "14": 0.01365713698018079, "15": 0.0138201993711368}, "unet": {"1": 0.029338991034390603, "2": 0.04098136983737797, "3": 0.053191183675822096, "4": 0.06900042770920613, "5": 0.08510530272953327, "6": 0.09226540044288743, "7": 0.11469343446647923, "8": 0.13317389850677966, "9": 0.14125110079062939, "10": 0.154387593461091, "11": 0.16638505648307758, "12": 0.19492023764178154, "13": 0.21799246336405093, "14": 0.2606973672859521, "15": 0.27655148591300605}, "vae": {"1": 0.03376312875399488, "2": 0.07399217916849121, "3": 0.11004193404680071, "4": 0.14354593551192943, "5": 0.17946397196879696, "6": 0.21222230604835574, "7": 0.24667370498105132, "8": 0.2812068222449894, "9": 0.31662720887997775, "10": 0.3520306087460141, "11": 0.39040729648728467, "12": 0.4260003667201866, "13": 0.47153294856477646, "14": 0.5068235329185159, "15": 0.5415168143338477}, "safety": {"1": 0.03063526479106153, "2": 0.04797388670138187, "3": 0.0672388456319694, "4": 0.08437921952686667, "5": 0.1039594023502502, "6": 0.12115485408823147, "7": 0.1391513596042759, "8": 0.16103985712797653, "9": 0.17952639864861664, "10": 0.22299047340615297, "11": 0.2438703968977699, "12": 0.27341513347404833, "13": 0.2970558778869773, "14": 0.31895873308754885, "15": 0.3440723231580761}}
        
        self.h100_profile_latency_256 = {"clip": {"1": 0.010763356614173675, "2": 0.011122808831327774, "3": 0.011640708931550687, "4": 0.011705784495764723, "5": 0.012307808962555565, "6": 0.01250345816797748, "7": 0.012666157484814828, "8": 0.012823822300719179, "9": 0.0126037277205258, "10": 0.012753783903540448, "11": 0.012932783448474204, "12": 0.013128956403345503, "13": 0.013358378706842053, "14": 0.012949418878027549, "15": 0.013127381081118229}, "unet": {"1": 0.0254171684156267, "2": 0.025889276108723516, "3": 0.026742454751261642, "4": 0.028399059777053034, "5": 0.02930485231003591, "6": 0.030402309508347998, "7": 0.031128080295664922, "8": 0.03151123231390909, "9": 0.032681762183807334, "10": 0.03391041530638325, "11": 0.03527436563174943, "12": 0.03681413127983711, "13": 0.03926054316059667, "14": 0.04135559788163828, "15": 0.04361406611088587}, "vae": {"1": 0.008543856308928557, "2": 0.013037726454132674, "3": 0.016666030511260033, "4": 0.01988933689664213, "5": 0.023583694494196346, "6": 0.026658064647748763, "7": 0.0303262718097896, "8": 0.03371112884915605, "9": 0.03733311519406888, "10": 0.040571440087289225, "11": 0.04389081437292756, "12": 0.047191714449804655, "13": 0.051138482797814876, "14": 0.0542412581378404, "15": 0.05792580714107168}, "safety": {"1": 0.027158761435017293, "2": 0.04097433093631146, "3": 0.053615434657937534, "4": 0.06755093234824017, "5": 0.08050565083976835, "6": 0.0936376547098768, "7": 0.10706905651992808, "8": 0.12038751328958476, "9": 0.13502245724278814, "10": 0.14927166088068106, "11": 0.1636664044468644, "12": 0.17882538171640286, "13": 0.1926385347751227, "14": 0.2070075457400464, "15": 0.2225177216529846}}
        self.h100_profile_latency_512 = {"clip": {"1": 0.010729043155300374, "2": 0.011114207970700701, "3": 0.01162113156169653, "4": 0.011685678735375404, "5": 0.01228585111675784, "6": 0.012452897919836094, "7": 0.012583264557179064, "8": 0.012737178839743138, "9": 0.012580370473466357, "10": 0.012718318921929978, "11": 0.012892252879635411, "12": 0.013105942669542545, "13": 0.013317528960047936, "14": 0.012893355878380438, "15": 0.013091435142773755}, "unet": {"1": 0.02640514894939807, "2": 0.03186587899999351, "3": 0.04127401993514932, "4": 0.04956639879288114, "5": 0.06105041731985248, "6": 0.06856303299233939, "7": 0.07873859376801799, "8": 0.08774662281696995, "9": 0.09964735091974337, "10": 0.10881113736346985, "11": 0.11614277953049168, "12": 0.12608531462804726, "13": 0.13460653038540235, "14": 0.14421168505214155, "15": 0.1532588206270399}, "vae": {"1": 0.025981670264534806, "2": 0.04397096617945603, "3": 0.05861139535067641, "4": 0.07297860169593169, "5": 0.08819514516817063, "6": 0.1027777564654849, "7": 0.11762863239843627, "8": 0.13209553646417904, "9": 0.14748582096459964, "10": 0.16170084267398532, "11": 0.17982623330792602, "12": 0.19494961027284058, "13": 0.21056599848504579, "14": 0.2290685014070376, "15": 0.24449221099897914}, "safety": {"1": 0.03995978668788258, "2": 0.06576067601175357, "3": 0.09061296585392445, "4": 0.11743271744732435, "5": 0.14353326063755215, "6": 0.16942864021628487, "7": 0.19585907678507888, "8": 0.2224162568260605, "9": 0.2503597050284346, "10": 0.3000266294173421, "11": 0.3281733137167369, "12": 0.359497005022907, "13": 0.39078635129393363, "14": 0.41930771552558455, "15": 0.45168329624736564}}
        self.batch_upper_bound = 0
        if profile_device == "h100":
            if image_size == 256:
                self.profile_latency = self.h100_profile_latency_256
                self.batch_upper_bound = 40
            elif image_size == 512:
                self.profile_latency = self.h100_profile_latency_512
                self.batch_upper_bound = 15
        elif profile_device == "rtx4090":
            if image_size == 256:
                self.profile_latency = self.rtx4090_profile_latency_256
                self.batch_upper_bound = 40
            elif image_size == 512:
                self.profile_latency = self.module_latency_rtx4090_512
                self.batch_upper_bound = 15

        self.instant_profile = {}
        self.instant_profile_trust = 999999 # an impossible value
        self.deploy_ready = deploy_ready
        self.extra_vae_safety_time = extra_vae_safety_time
        for batch_size in range(self.batch_upper_bound+1):
            self.instant_profile[batch_size] = {}
            self.instant_profile[batch_size]["count"] = 0
            self.instant_profile[batch_size]["average_latency"] = 0

        self.total_request_count = 0
        self.goodput = 0
        self.log_prefix = log_prefix
        if "device" in kwargs:
            self.device = kwargs["device"]
            for stream_module in self.stream_module_list:
                stream_module.device = self.device
        else:
            self.device = "cuda" # default to cuda:0

        self.step_slo_scale = step_slo_scale
        self.step_delta = step_delta
        # one step slo, controlled by two values
        self.slo_batch_size = 0
        print(f"yhc debug:: step_factor = {step_slo_scale * step_delta}")

        if self.id == "clip":
            self.step_slo_latency = self.profile_latency["clip"]["1"] * self.step_slo_scale * self.step_delta
            while self.profile_latency["clip"][str(self.slo_batch_size + 1)] < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        elif self.id == "unet":
            self.step_slo_latency = self.profile_latency["unet"]["1"] * self.step_slo_scale * self.step_delta
            while self.profile_latency["unet"][str(self.slo_batch_size + 1)] < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        elif self.id == "vae&safety":
            self.step_slo_latency = (self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]) * self.step_slo_scale * self.step_delta
            while (self.profile_latency["vae"][str(self.slo_batch_size + 1)] + self.profile_latency["safety"][str(self.slo_batch_size + 1)])< self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        print(f"yhc debug:: hold module:{self.module_tag}, slo_batch_size={self.slo_batch_size}")
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
        while True:
            # put request to batch_queue
            while not self.input_queue.empty():
                request = self.input_queue.get()
                if request == None:
                    print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, received terminate signal!")
                    self.output_queue.put(None)
                    self.terminate_receive_flag = True
                    if self.id == "unet":
                        batch_latency_record = open(f"{self.log_prefix}_unet_batch_latency.log", "w")
                        json.dump(self.instant_profile, batch_latency_record)
                    break

                # log the request info
                if self.id == "unet":
                    info ={
                        "request_time": request["request_time"],
                        "id": request["id"],
                        "SLO": request["SLO"]
                    }
                    self.log_file.write(json.dumps(info)+"\n")

                self.total_request_count += 1
                request[self.module_tag+"_receive_time"] = time.time()
                # check whether can serve
                if self.id == "clip":
                    if request["SLO"] + request["request_time"] < time.time() + self.profile_latency["clip"]["1"] + self.profile_latency["unet"]["1"] * request["loop_num"]["UNetModule"] + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        self.log_file.write(info+"\n")
                        print(info)
                    else:
                        # check tensor whether on same device
                        for key in request.keys():
                            if type(request[key]) == torch.Tensor:
                                request[key] = request[key].to(self.device)
                        self.batch_queue.put(ComparableRequest(time.time(), request))
                elif self.id == "unet":
                    if request["SLO"] + request["request_time"] < time.time() + self.profile_latency["unet"]["1"] * request["loop_num"]["UNetModule"] + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        self.log_file.write(info+"\n")
                        print(info)
                    else:
                        for key in request.keys():
                            if type(request[key]) == torch.Tensor:
                                request[key] = request[key].to(self.device)
                        request["wait_loop_count"] = 0
                        self.batch_queue.put(ComparableRequest(time.time(), request))
                elif self.id == "vae&safety":
                    if request["SLO"] + request["request_time"] < time.time() + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        self.log_file.write(info+"\n")
                        print(info)
                        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, goodput rate: {self.goodput / self.total_request_count}, goodput: {self.goodput}, total_request: {self.total_request_count}")
                    else:
                        # check tensor whether on same device
                        for key in request.keys():
                            if type(request[key]) == torch.Tensor:
                                request[key] = request[key].to(self.device)
                        self.batch_queue.put(ComparableRequest(time.time(), request))
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
                        for loop_module in self.loop_module_list:
                            for item in self.current_batch:
                                if item.data["loop_index"][loop_module] >= item.data["loop_num"][loop_module]:
                                    item.data[self.module_tag+"_send_time"] = time.time()
                                    for key in item.data.keys():
                                        if type(item.data[key]) == torch.Tensor:
                                            item.data[key] = item.data[key].cpu()
                                    self.output_queue.put(item.data)
                                else:
                                    # 没跑够loop_count，重新回queue
                                    self.batch_queue.put(item)
                    # no loop module
                    else:
                        #self.total_request_count += len(self.current_batch)
                        for item in self.current_batch:
                            item.data[self.module_tag+"_send_time"] = time.time()
                            if time.time() <= item.data["SLO"] + item.data["request_time"]:
                                self.goodput += 1
                            # tensor have to be moved to cpu when passing across device
                                for key in item.data.keys():
                                    if type(item.data[key]) == torch.Tensor:
                                        item.data[key] = item.data[key].cpu()
                                self.output_queue.put(item.data)
                        info = f"goodput: {self.goodput}, total_request: {self.total_request_count}\n"
                        self.log_file.write(info)
                        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, goodput rate: {self.goodput / self.total_request_count}, goodput: {self.goodput}, total_request: {self.total_request_count}")
                #self.log_file.flush()
            scatter()

            # 将batch_queue里所有超时的request都扔掉
            if self.id == "unet":
                valid_request_list = []
                while not self.batch_queue.empty():
                    item = self.batch_queue.get()
                    # 超时则丢弃
                    # 尝试用过去run的batch信息来profile
                    if self.instant_profile[1]["count"] >= self.instant_profile_trust:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.instant_profile[1]["average_latency"] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                    # instant_profile次数比较少，不采纳
                    else:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.profile_latency["unet"]["1"] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
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
                    self.batch_queue.put(item)

                valid_request_list = []
                while not self.low_priority_batch_queue.empty():
                    item = self.low_priority_batch_queue.get()
                    # 超时则丢弃
                    # 尝试用过去run的batch信息来profile
                    if self.instant_profile[1]["count"] >= self.instant_profile_trust:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.instant_profile[1]["average_latency"] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                    # instant_profile次数比较少，不采纳
                    else:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.profile_latency["unet"]["1"] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.profile_latency["vae"]["1"] + self.profile_latency["safety"]["1"]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                # 有效的request放回batch_queue
                for item in valid_request_list:
                    # 更新priority
                    # 不更新priority，避免出现交替schedule
                    self.low_priority_batch_queue.put(item)
            # 及时写回
            #self.log_file.flush()
            
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

            self.schedule_end = time.time()
            self.batch_ready.release()
            self.log_file.write(f"")
            self.log_file.flush()

    @torch.no_grad()
    def run(self, **kwargs):
        log_file = open(f"{self.log_prefix}_{self.module_tag}.log", "w")
        self.log_file = log_file
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
                    # write the batch log

                    execution_begin = time.perf_counter()

                    # wipe off the priority
                    # below list just passed reference, not data copy
                    batch_request = [item.data for item in self.current_batch]
                    # execute through the pipeline
                    if self.id == "unet":
                        for _ in range(1):
                            batch_request = self.stream_module_list[0].compute(batch_request)
                            # update loop_index
                            for request in batch_request:
                                request["loop_index"][self.loop_module_list[0]] += 1
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
                    # calculate the new batch latency
                    if self.id == "unet":
                        profile_latency = time.perf_counter() - execution_begin
                        current_batch_size = len(self.current_batch)
                    
                        new_average_latency =  (self.instant_profile[current_batch_size]["average_latency"] * self.instant_profile[current_batch_size]["count"] + profile_latency) / (self.instant_profile[current_batch_size]["count"] + 1)
                        self.instant_profile[current_batch_size]["average_latency"] = new_average_latency
                        self.instant_profile[current_batch_size]["count"] += 1
                        #self.log_file.write(json.dumps(self.instant_profile)+"\n")"""
                    self.batch_used.release()
                
            # Worker process receive interrupt
            except KeyboardInterrupt:
                print("-"*10, f"Worker process:[{os.getpid()}] received KeyboardInterrupt.", "-"*10)