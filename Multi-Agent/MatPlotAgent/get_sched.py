log_file = ['codellama_1.log',"llava_1.log","llava_2.log", "codellama_2.log"]

def parse_log(file_name):
    time_summary = []
    index = 0
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "pred_time_list:" in line:
                index = line.index("pred_time_list:") + len("pred_time_list:")
                time_list = eval(line[index:])
                clean_time_list = [x for x in time_list if x[0] != 0]
                time_summary.extend(clean_time_list)
    return time_summary
all_time = []
for log in log_file:
    all_time.extend(parse_log("./log/"+log))
all_time = all_time[:]
aver_rate = sum([x[0]/x[1] for x in all_time])/len(all_time)
print(f"average_rate: {round(aver_rate * 100, 2)} %")
"""
average_rate: 0.05311554027988682
0.0531
"""