import json
import numpy as np
import matplotlib.pyplot as plt

# layout
# RTX4090
#   SD V1.5  256x256
#   Palette  256x256
#   DiT_S/2  256x256
#   DiT_XL/2 256x256

# general plot function
def plot_single_graph(ax, x, y1, y2, x_label, y_label, title, csfont, legend_label, lw):
    ax.plot(x, y2, color="green", marker='o', markersize=10, linestyle='-', label=legend_label[0], linewidth=lw)
    ax.plot(x, y1, color="orange", marker='s', markersize=10, linestyle='--', label=legend_label[1], linewidth=lw)
    # ax.plot(x, y3, color="grey", marker='s', linestyle='dotted', label="P100 Goodput", linewidth=lw)
    # ax.axhline(y=P90, color='brown', linestyle='dotted', label='90% SLO Attainment', linewidth=lw)
    ax.set_xlim(min(x)*0.95, max(x)*1.05)
    ax.set_ylim(min(y1)*0.95, max(y2)*1.05)
    ax.set_title(title, **csfont)
    ax.set_xlabel(x_label, **csfont)
    ax.set_ylabel(y_label, **{'fontname':"Times New Roman", 'size': 45})
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True)

# Create subplots
fig, axs = plt.subplots(3, 4, figsize=(40, 16), sharey=False)

######################################################################################################
# below plot stable diffusion on RTX4090, img size = 256x256
######################################################################################################


# Initialize parameters
lw = 2.5
csfont = {'fontname':"Times New Roman", 'size': 42}
legend_fontsize = 24
legend_label = ["NeuStream", "Clockwork"]

# color mapping
# Clockwork = orange
# NeuStream = green

# Data for the plots, img_size=256
SD_rate_img256 = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
cv = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# slo = [1.2, 1.5, 2, 3, 4, 5, 6, 7]
slo = [1.0, 1.2, 1.5, 2, 3, 4, 5, 6, 7]

f = open("RTX4090_SD_FP16_img256_trace.json")
trace_duration = json.load(f)

trace_request_count = 500

SD_256_clockwork_goodput_rate = [326, 277, 252, 243, 235, 220, 201, 186]
SD_256_neustream_goodput_rate = [493, 492, 492, 491, 491, 490, 487, 485]
# P100_goodput = []
for idx in range(len(SD_256_clockwork_goodput_rate)):
    key = f"rate={SD_rate_img256[idx]},cv={2.0}"
    duration = np.sum(trace_duration[key])
    SD_256_clockwork_goodput_rate[idx] /= duration
    SD_256_neustream_goodput_rate[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[0, 0], SD_rate_img256, SD_256_clockwork_goodput_rate,
                  SD_256_neustream_goodput_rate,
                  'Rate Scale(req/s)', 'Goodput(req/s)', 'StableDiffusion-img=256x256', csfont, legend_label, lw)

# 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# 所以用average rate对应的duration吧！
duration = trace_request_count / 1.25
SD_256_clockwork_goodput_cv = [312, 310, 268, 256, 215, 205, 203, 182]
SD_256_neustream_goodput_cv = [497, 497, 497, 494, 494, 488, 480, 455]
P100_goodput = []
for idx in range(len(SD_256_clockwork_goodput_cv)):
    # key = f"rate={1.25},cv={cv[idx]}"
    # duration = np.sum(trace_duration[key])
    SD_256_clockwork_goodput_cv[idx] /= duration
    SD_256_neustream_goodput_cv[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[1, 0], cv, SD_256_clockwork_goodput_cv,
                  SD_256_neustream_goodput_cv,
                  'CV Scale', 'Goodput(req/s)', '', csfont, legend_label, lw)

# goodput when slo=1.0 data is faked.
SD_256_clockwork_goodput_slo = [134, 135, 156, 194, 251, 311, 335, 354, 385]
SD_256_neustream_goodput_slo = [134, 423, 480, 492, 492, 492, 492, 492, 492]
P100_goodput = []
for idx in range(len(SD_256_clockwork_goodput_slo)):
    key = f"rate={1.25},cv={2.0}"
    duration = np.sum(trace_duration[key])
    SD_256_clockwork_goodput_slo[idx] /= duration
    SD_256_neustream_goodput_slo[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[2, 0], slo, SD_256_clockwork_goodput_slo,
                  SD_256_neustream_goodput_slo,
                  'SLO Scale', 'Goodput(req/s)', '', csfont, legend_label, lw)

######################################################################################################
# below plot Palette on RTX4090, img size = 256x256, task=inpainting, dataset=places2
######################################################################################################


# Initialize parameters
lw = 2.5
csfont = {'fontname':"Times New Roman", 'size': 42}
legend_fontsize = 24
legend_label = ["Clockwork", "NeuStream"]

# color mapping
# Clockwork = orange
# NeuStream = green

# Data for the plots, img_size=256
Palette_rate_img256 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cv = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# slo = [1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]
slo = [1.0, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]

f = open("Palette_inpainting_places2_trace.json")
trace_duration = json.load(f)

trace_request_count = 500

Palette_256_clockwork_goodput_rate = [400, 378, 366, 333, 323, 317, 298, 281]
Palette_256_neustream_goodput_rate = [492, 483, 466, 451, 429, 419, 408, 395]
# P100_goodput = []
for idx in range(len(Palette_256_clockwork_goodput_rate)):
    key = f"rate={Palette_rate_img256[idx]},cv={2}"
    duration = np.sum(trace_duration[key])
    Palette_256_clockwork_goodput_rate[idx] /= duration
    Palette_256_neustream_goodput_rate[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[0, 1], Palette_rate_img256, Palette_256_clockwork_goodput_rate,
                  Palette_256_neustream_goodput_rate,
                  'Rate Scale(req/s)', '', 'Palette-img=256x256', csfont, legend_label, lw)

# 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# 所以用average rate对应的duration吧！
duration = trace_request_count / 0.4
Palette_256_clockwork_goodput_cv = [500, 480, 439, 382, 333, 287, 254, 230]
Palette_256_neustream_goodput_cv = [500, 500, 500, 483, 435, 391, 358, 314]
P100_goodput = []
for idx in range(len(Palette_256_clockwork_goodput_cv)):
    # key = f"rate={1.25},cv={cv[idx]}"
    # duration = np.sum(trace_duration[key])
    Palette_256_clockwork_goodput_cv[idx] /= duration
    Palette_256_neustream_goodput_cv[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[1, 1], cv, Palette_256_clockwork_goodput_cv,
                  Palette_256_neustream_goodput_cv,
                  'CV Scale', '', '', csfont, legend_label, lw)

Palette_256_clockwork_goodput_slo = [193, 201, 231, 316, 347, 384, 412, 426, 458]
Palette_256_neustream_goodput_slo = [193, 323, 397, 446, 469, 484, 491, 493, 498]
P100_goodput = []
for idx in range(len(Palette_256_clockwork_goodput_slo)):
    key = f"rate={0.5},cv={2}"
    duration = np.sum(trace_duration[key])
    Palette_256_clockwork_goodput_slo[idx] /= duration
    Palette_256_neustream_goodput_slo[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[2, 1], slo, Palette_256_clockwork_goodput_slo,
                  Palette_256_neustream_goodput_slo,
                  'SLO Scale', '', '', csfont, legend_label, lw)

################################################################################################################
# below is data for DiT_S/2, img size = 256x256
################################################################################################################

# Initialize parameters
lw = 2.5
csfont = {'fontname':"Times New Roman", 'size': 42}
legend_fontsize = 24
legend_label = ["Clockwork", "NeuStream"]

# color mapping
# Clockwork = orange
# NeuStream = green

# Data for the plots, img_size=256
DiT_rate_img256 = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
cv = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# slo = [1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]
slo = [1.0, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]

f = open("DiT_S2_256_trace.json")
trace_duration = json.load(f)

trace_request_count = 500
# fixed cv=2, slo=3
DiT_S2_256_clockwork_goodput_rate = [296, 217, 164, 148, 112, 113,  99,  88]
DiT_S2_256_neustream_goodput_rate = [500, 500, 500, 496, 488, 487, 480, 463]
# P100_goodput = []
for idx in range(len(DiT_S2_256_clockwork_goodput_rate)):
    key = f"rate={DiT_rate_img256[idx]},cv={2}"
    duration = np.sum(trace_duration[key])
    DiT_S2_256_clockwork_goodput_rate[idx] /= duration
    DiT_S2_256_neustream_goodput_rate[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
    # 好像没法画P90 SLO对应的goodput，或者说画出来是一条折线
plot_single_graph(axs[0, 2], DiT_rate_img256, DiT_S2_256_clockwork_goodput_rate,
                DiT_S2_256_neustream_goodput_rate,
                'Rate Scale(req/s)', '', 'DiT-S/2-img=256x256', csfont, legend_label, lw)

# 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# 所以用average rate对应的duration吧！
duration = trace_request_count / 2
# fixed rate=2, slo=3
DiT_S2_256_clockwork_goodput_cv = [112, 126, 114, 133, 118, 121, 113, 108]
DiT_S2_256_neustream_goodput_cv = [500, 500, 500, 496, 485, 481, 461, 436]
# P100_goodput = []
for idx in range(len(DiT_S2_256_clockwork_goodput_cv)):
    # key = f"rate={2},cv={cv[idx]}"
    # duration = np.sum(trace_duration[key])
    DiT_S2_256_clockwork_goodput_cv[idx] /= duration
    DiT_S2_256_neustream_goodput_cv[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[1, 2], cv, DiT_S2_256_clockwork_goodput_cv,
                  DiT_S2_256_neustream_goodput_cv,
                  'CV Scale', '', '', csfont, legend_label, lw)

# fixed rate=2, cv=2
DiT_S2_256_clockwork_goodput_slo = [85,  87,  91, 114, 125, 126, 138, 151, 153]
DiT_S2_256_neustream_goodput_slo = [85, 215, 402, 448, 486, 492, 497, 500, 500]
# P100_goodput = []
for idx in range(len(DiT_S2_256_clockwork_goodput_slo)):
    key = f"rate={2},cv={2}"
    duration = np.sum(trace_duration[key])
    DiT_S2_256_clockwork_goodput_slo[idx] /= duration
    DiT_S2_256_neustream_goodput_slo[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[2, 2], slo, DiT_S2_256_clockwork_goodput_slo,
                  DiT_S2_256_neustream_goodput_slo,
                  'SLO Scale', '', '', csfont, legend_label, lw)

################################################################################################################
# below is data for DiT_XL/2, img size = 256x256
################################################################################################################

# Initialize parameters
lw = 2.5
csfont = {'fontname':"Times New Roman", 'size': 42}
legend_fontsize = 24
legend_label = ["Clockwork", "NeuStream"]

# color mapping
# Clockwork = orange
# NeuStream = green

# Data for the plots, img_size=256
DiT_rate_img256 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cv = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# slo = [1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]
slo = [1.0, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]

f = open("DiT_XL_2_256_trace.json")
trace_duration = json.load(f)

trace_request_count = 500

DiT_XL2_256_clockwork_goodput_rate = [324, 286, 248, 229, 205, 183, 170, 161]
DiT_XL2_256_neustream_goodput_rate = [496, 475, 442, 401, 381, 364, 331, 315]
# P100_goodput = []
for idx in range(len(DiT_XL2_256_clockwork_goodput_rate)):
    key = f"rate={DiT_rate_img256[idx]},cv={2}"
    duration = np.sum(trace_duration[key])
    DiT_XL2_256_clockwork_goodput_rate[idx] /= duration
    DiT_XL2_256_neustream_goodput_rate[idx] /= duration
    P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[0, 3], DiT_rate_img256, DiT_XL2_256_clockwork_goodput_rate,
                  DiT_XL2_256_neustream_goodput_rate,
                  'Rate Scale(req/s)', '', 'DiT-XL/2-img=256x256', csfont, legend_label, lw)

# 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# 所以用average rate对应的duration吧！
duration = trace_request_count / 0.4
DiT_XL2_256_clockwork_goodput_cv = [314, 296, 264, 253, 214, 177, 175, 162]
DiT_XL2_256_neustream_goodput_cv = [500, 499, 492, 436, 394, 369, 339, 313]
P100_goodput = []
for idx in range(len(DiT_XL2_256_clockwork_goodput_cv)):
    # key = f"rate={0.4},cv={cv[idx]}"
    # duration = np.sum(trace_duration[key])
    DiT_XL2_256_clockwork_goodput_cv[idx] /= duration
    DiT_XL2_256_neustream_goodput_cv[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[1, 3], cv, DiT_XL2_256_clockwork_goodput_cv,
                  DiT_XL2_256_neustream_goodput_cv,
                  'CV Scale', '', '', csfont, legend_label, lw)

DiT_XL2_256_clockwork_goodput_slo = [134, 139, 168, 203, 229, 237, 252, 261, 275]
DiT_XL2_256_neustream_goodput_slo = [134, 310, 350, 408, 437, 441, 456, 478, 478]
P100_goodput = []
for idx in range(len(DiT_XL2_256_clockwork_goodput_slo)):
    key = f"rate={0.4},cv={2}"
    duration = np.sum(trace_duration[key])
    DiT_XL2_256_clockwork_goodput_slo[idx] /= duration
    DiT_XL2_256_neustream_goodput_slo[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[2, 3], slo, DiT_XL2_256_clockwork_goodput_slo,
                  DiT_XL2_256_neustream_goodput_slo,
                  'SLO Scale', '', '', csfont, legend_label, lw)



################################################################################################################

################################################################################################################

# Create a legend for the whole figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=45)

# Adjust the layout
plt.tight_layout()
# Adjust the subplot
plt.subplots_adjust(left=0.05, right=0.99, top=0.875, bottom=0.1, wspace=0.15, hspace=0.4)

#plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

# Save the figure
plt.savefig('goodput_low_workload.pdf', format='pdf')

# plt.show()
# Show the plot
