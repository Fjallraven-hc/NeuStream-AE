import re

workspace = "./log/"

pattern_pre = r'Goodprefill: ((-)?\d+\.\d+) req/s'
pattern_decode = r'Gooddecode: ((-)?\d+\.\d+) req/s'

def parse_goodput(file_name: str, pattern: str) :
    gpt_vllm = []
    i = 0
    with open(workspace + file_name, 'r') as f:
        lines = list(f.readlines())
    for line in lines:
        match = re.search(pattern, line)
        if match:
            gpt_vllm.append(float(match.group(1)))
            i += 1
    return gpt_vllm

file_name = ["codellama_1.log", "llava_1.log", "llava_2.log", "codellama_2.log"]

prefill_list = []
decode_list = []

for f in file_name:
    prefill_list.append(parse_goodput(f, pattern_pre))
    decode_list.append(parse_goodput(f, pattern_decode))

def add_two_llava(ll: list):
    for index in range(len(ll[1])):
        ll[1][index] = round(ll[2][index]+ll[1][index], 2)
    ll.pop(2)
    return ll

def rou(data: list[list[float]]):
    d = data[2]
    return round(sum(d)/len(d),2)

prefill_list = add_two_llava(prefill_list)
decode_list = add_two_llava(decode_list)    

vllm_p = []
vllm_d = []
ours_p = []
ours_d = []

exp = [5,4,4]

for idx, gpt in enumerate(prefill_list):
    if idx == 0:
        gpt.pop(85)
        gpt.pop(77)
        gpt = gpt[:26] + gpt[43:]
    else:
        gpt = gpt[:26] + gpt[42:]
    # print(len(gpt))
    v = [[] for _ in range(sum(exp))]
    n = [[] for _ in range(sum(exp))]
    for idx, g in enumerate(gpt):
        real_idx = (idx // 2) % (sum(exp))
        if idx % 2 == 0:
            n[real_idx].append(g)
        else:
            v[real_idx].append(g)
    index = 0
    for e in exp[:1]:
        # print(rou(n[index:index+e]))
        ours_p.append(rou(n[index:index+e]))
        vllm_p.append(rou(v[index:index+e]))
        index += e

for idx, gpt in enumerate(decode_list):
    if idx == 0:
        gpt.pop(85)
        gpt.pop(77)
        gpt = gpt[:26] + gpt[43:]
    else:
        gpt = gpt[:26] + gpt[42:]
    v = [[] for _ in range(sum(exp))]
    n = [[] for _ in range(sum(exp))]
    for idx, g in enumerate(gpt):
        real_idx = (idx // 2) % (sum(exp))
        if idx % 2 == 0:
            n[real_idx].append(g)
        else:
            v[real_idx].append(g)
    index = 0
    for e in exp[:1]:
        ours_d.append(rou(n[index:index+e]))
        vllm_d.append(rou(v[index:index+e]))
        index += e

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
slo_labels = ['Code Llama 1', 'Llava', 'Code Llama 2']   # X 轴的 SLO 标签

# X 轴位置
x = np.arange(len(slo_labels))

# 条形的宽度
bar_width = 0.15
semi_bar = bar_width/2
gap = 0.05

# 绘制条形
fig, ax = plt.subplots(figsize=(15, 7))
bar1 = ax.bar(x - 3*semi_bar - gap, vllm_p, width=bar_width, label='vLLM p', hatch='//', edgecolor='blue', fill=False)
bar2 = ax.bar(x - 1*semi_bar - gap, vllm_d, width=bar_width, label='vLLm d', hatch='xx', edgecolor='orange', fill=False)
bar3 = ax.bar(x + 1*semi_bar + gap, ours_p, width=bar_width, label='Neu p', hatch='\\\\', edgecolor='green', fill=False)
bar4 = ax.bar(x + 3*semi_bar + gap, ours_d, width=bar_width, label='Neu d', hatch='xx', edgecolor='red', fill=False)

# 添加图例
ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(0.85, 1)) 

# X 轴和 Y 轴标签
ax.set_xlabel('Model Type', size=24)
ax.set_ylabel('Normalized Goodput Rate', size=24)
# ax.set_title('Comparison of Systems', size=24)
ax.set_xticks(x)
ax.set_xticklabels(slo_labels,fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_ylim(0, 0.8)

# 调整 Y 轴范围

# 添加垂直网格线，设置为灰色的虚线
ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

# 显示图形
plt.tight_layout()
plt.savefig("norm_goodput_revised.pdf", format="pdf")
plt.show()