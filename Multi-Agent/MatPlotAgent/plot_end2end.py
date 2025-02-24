import re

workspace = "./log/"

pattern_goodput = r'Goodput: ((-)?\d+\.\d+) tokens/s'

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

gpt_list = []

for f in file_name:
    gpt_list.append(parse_goodput(f, pattern_goodput))


def add_two_llava(ll: list):
    for index in range(len(ll[1])):
        ll[1][index] = round(ll[2][index]+ll[1][index], 2)
    ll.pop(2)
    return ll

def rou(data: list[list[float]]):
    return "np.array(" + str([round(sum(d)/len(d), 2) for d in data]) + ")"

gpt_list = add_two_llava(gpt_list)    

exp = [5,4,4]
goodput_figure = []
for idx, gpt in enumerate(gpt_list):
    v = [[] for _ in range(sum(exp))]
    n = [[] for _ in range(sum(exp))]
    for idx, g in enumerate(gpt):
        real_idx = (idx // 2) % (sum(exp))
        if idx % 2 == 0:
            n[real_idx].append(g)
        else:
            v[real_idx].append(g)
    index = 0
    for e in exp:
        goodput_figure.append(rou(n[index:index+e]))
        goodput_figure.append(rou(v[index:index+e]))
        index += e

import numpy as np
import matplotlib.pyplot as plt

def plot_single_graph(ax, x, y1, y2, x_label, y_label, title, csfont, legend_label, lw):
    ax.plot(x, y1, color="green", marker='o', linestyle='-', label=legend_label[0], linewidth=lw, markersize=13.5)
    ax.plot(x, y2, color="orange", marker='s', linestyle='--', label=legend_label[1], linewidth=lw, markersize=13.5)
    ax.set_xlim(x[0]*0.95, x[-1]*1.05)
    #ax.set_ylim(0, 101)
    ax.set_title(title, **{'fontname':"Times New Roman", 'size': 42})#**csfont)
    ax.set_xlabel(x_label, **{'fontname':"Times New Roman", 'size': 56})#**csfont)
    if y_label is not None:
        ax.set_ylabel(y_label, **{'fontname':"Times New Roman", 'size': 48})
    ax.tick_params(axis='both', labelsize=36)
    ax.grid(True)


# Initialize parameters
lw = 3.375
csfont = {'fontname':"Times New Roman", 'size': 42}
legend_fontsize = 24
legend_label = ["NeuStream", "vLLM"]

fig, axs = plt.subplots(3, 3, figsize=(27, 27), sharey=False)


rate = np.array([0.6,0.7,0.8,0.9,1.0])
o_gpt_vsrate = eval(goodput_figure[0])
v_gpt_vsrate = eval(goodput_figure[1])

cv = np.array([1, 2, 3, 4])
o_gpt_vscv = eval(goodput_figure[2])
v_gpt_vscv = eval(goodput_figure[3])

dslo = np.array([1.2, 1.4, 1.6, 1.8])
o_gpt_vsslo = eval(goodput_figure[4])
v_gpt_vsslo = eval(goodput_figure[5])

plot_single_graph(axs[0, 0], rate, o_gpt_vsrate, v_gpt_vsrate,
                  'Rate Scale(req/s)', 'Goodput (token/s)', 'Code Llama 1', csfont, legend_label, lw)

plot_single_graph(axs[1, 0], cv, o_gpt_vscv, v_gpt_vscv,
                  'CV Scale', 'Goodput (token/s)', '', csfont, legend_label, lw)

plot_single_graph(axs[2, 0], dslo, o_gpt_vsslo, v_gpt_vsslo,
                  'SLO Scale', 'Goodput (token/s)', '', csfont, legend_label, lw)


o_gpt_vsrate = eval(goodput_figure[6])
v_gpt_vsrate = eval(goodput_figure[7])

# cv
# cv = np.array([1, 1.5, 2, 3, 4])
o_gpt_vscv = eval(goodput_figure[8])
v_gpt_vscv = eval(goodput_figure[9])

# slo
# pslo=1.5, pslo的影响不大
# dslo = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
o_gpt_vsslo = eval(goodput_figure[10])
v_gpt_vsslo = eval(goodput_figure[11])

plot_single_graph(axs[0, 1], rate, o_gpt_vsrate, v_gpt_vsrate,
                  'Rate Scale(req/s)', None, 'LLaVA', csfont, legend_label, lw)

plot_single_graph(axs[1, 1], cv, o_gpt_vscv, v_gpt_vscv,
                  'CV Scale', None, '', csfont, legend_label, lw)

plot_single_graph(axs[2, 1], dslo, o_gpt_vsslo, v_gpt_vsslo,
                  'SLO Scale', None, '', csfont, legend_label, lw)



o_gpt_vsrate = eval(goodput_figure[12])
v_gpt_vsrate = eval(goodput_figure[13])
# cv
# LLaVA/log/log-0414-1-ours-rate1.5-cv1-4-predstep30-nooc_3000reqs
# cv = np.array([1, 1.5, 2, 3, 4])
o_gpt_vscv = eval(goodput_figure[14])
v_gpt_vscv = eval(goodput_figure[15])

o_gpt_vsslo = eval(goodput_figure[16])
v_gpt_vsslo = eval(goodput_figure[17])
plot_single_graph(axs[0, 2], rate, o_gpt_vsrate, v_gpt_vsrate,
                  'Rate Scale(req/s)', None, 'Code Llama 2', csfont, legend_label, lw)

plot_single_graph(axs[1, 2], cv, o_gpt_vscv, v_gpt_vscv,
                  'CV Scale', None, '', csfont, legend_label, lw)

plot_single_graph(axs[2, 2], dslo, o_gpt_vsslo, v_gpt_vsslo,
                  'SLO Scale', None, '', csfont, legend_label, lw)

plt.tight_layout()

plt.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.1, wspace=0.15, hspace=0.4)

# Create a legend for the whole figure
handles, labels = axs[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=50)

# Save the figure
plt.savefig('./agent_end2end_4090.pdf', format='pdf')
