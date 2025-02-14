import json
import numpy as np
import matplotlib.pyplot as plt

# layout
# RTX4090
# H100
#   SD V1.5  512x512
#   DiT_S/2  512x512

# general plot function
def plot_single_graph(ax, x, y1, y2, x_label, y_label, title, csfont, legend_label, lw, ylim=None):
    ax.plot(x, y2, color="green", marker='o', markersize=10, linestyle='-', label=legend_label[0], linewidth=lw)
    ax.plot(x, y1, color="orange", marker='s', markersize=10, linestyle='--', label=legend_label[1], linewidth=lw)
    # ax.plot(x, y3, color="grey", marker='s', linestyle='dotted', label="P100 Goodput", linewidth=lw)
    # ax.axhline(y=P90, color='brown', linestyle='dotted', label='90% SLO Attainment', linewidth=lw)
    ax.set_xlim(min(x)*0.95, max(x)*1.05)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(min(y1)*0.95, max(y2)*1.05)
    ax.set_title(title, **csfont)
    ax.set_xlabel(x_label, **csfont)
    ax.set_ylabel(y_label, **{'fontname':"Times New Roman", 'size': 45})
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True)

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(25, 16), sharey=False)

######################################################################################################
# below plot stable diffusion on RTX4090, img size = 512x512, RTX4090
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
SD_rate_img512 = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55] 
cv = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
slo = [1.2, 1.5, 2, 3, 4, 5, 6, 7]

f = open("RTX4090_SD_FP16_img512_trace.json")
trace_duration = json.load(f)

trace_request_count = 500

SD_512_clockwork_goodput_rate = [415, 391, 371, 369, 369, 345, 344, 337]
SD_512_neustream_goodput_rate = [468, 460, 445, 433, 422, 415, 402, 388]
P100_goodput = []
for idx in range(len(SD_512_clockwork_goodput_rate)):
    key = f"rate={SD_rate_img512[idx]},cv={2.0}"
    duration = np.sum(trace_duration[key])
    SD_512_clockwork_goodput_rate[idx] /= duration
    SD_512_neustream_goodput_rate[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[0, 0], SD_rate_img512, SD_512_clockwork_goodput_rate,
                  SD_512_neustream_goodput_rate,
                  'Rate Scale', 'Goodput(req/s)', 'StableDiffusion-img=512x512-RTX4090', 
                  csfont, legend_label, lw, ylim=(0, 1.2))

# 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# 所以用average rate对应的duration吧！
duration = trace_request_count / 0.3
SD_512_clockwork_goodput_cv = [500, 493, 457, 401, 338, 285, 245, 218]
SD_512_neustream_goodput_cv = [498, 499, 488, 443, 375, 342, 287, 255]
P100_goodput = []
for idx in range(len(SD_512_clockwork_goodput_cv)):
    # key = f"rate={0.3},cv={cv[idx]}"
    # duration = np.sum(trace_duration[key])
    SD_512_clockwork_goodput_cv[idx] /= duration
    SD_512_neustream_goodput_cv[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[1, 0], cv, SD_512_clockwork_goodput_cv,
                  SD_512_neustream_goodput_cv,
                  'CV Scale', 'Goodput(req/s)', '', 
                  csfont, legend_label, lw, ylim=(0, 1.3))

SD_512_clockwork_goodput_slo = [220, 237, 306, 402, 450, 471, 483, 493]
SD_512_neustream_goodput_slo = [220, 251, 364, 449, 481, 488, 489, 493]
# P100_goodput = []
for idx in range(len(SD_512_clockwork_goodput_slo)):
    key = f"rate={0.3},cv={2.0}"
    duration = np.sum(trace_duration[key])
    SD_512_clockwork_goodput_slo[idx] /= duration
    SD_512_neustream_goodput_slo[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[2, 0], slo, SD_512_clockwork_goodput_slo,
                  SD_512_neustream_goodput_slo,
                  'SLO Scale', 'Goodput(req/s)', '', 
                  csfont, legend_label, lw, ylim=(0, 1.2))

######################################################################################################
# below plot stable diffusion on H100, img size = 512x512, H100
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
SD_rate_img512 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
cv = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
slo = [1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

f = open("H100_SD_FP16_img512_trace.json")
trace_duration = json.load(f)

trace_request_count = 500

SD_512_clockwork_goodput_rate = [297, 211, 237, 219, 220, 199, 207]
SD_512_neustream_goodput_rate = [431, 398, 377, 354, 334, 322, 313]
# P100_goodput = []
for idx in range(len(SD_512_clockwork_goodput_rate)):
    key = f"rate={SD_rate_img512[idx]},cv={2.0}"
    duration = np.sum(trace_duration[key])
    SD_512_clockwork_goodput_rate[idx] /= duration
    SD_512_neustream_goodput_rate[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[0, 1], SD_rate_img512, SD_512_clockwork_goodput_rate,
                  SD_512_neustream_goodput_rate,
                  'Rate Scale', '', 'StableDiffusion-img=512x512-H100', 
                  csfont, legend_label, lw, ylim=(0, 1.2))

SD_512_clockwork_goodput_cv = [488, 386, 320, 292, 197, 197, 180]
SD_512_neustream_goodput_cv = [497, 498, 497, 466, 375, 301, 263]
# P100_goodput = []

# 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# 所以用average rate对应的duration吧！
duration = trace_request_count / 1.25
for idx in range(len(SD_512_clockwork_goodput_cv)):
    # key = f"rate={1.25},cv={cv[idx]}"
    # duration = np.sum(trace_duration[key])
    SD_512_clockwork_goodput_cv[idx] /= duration
    SD_512_neustream_goodput_cv[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[1, 1], cv, SD_512_clockwork_goodput_cv,
                  SD_512_neustream_goodput_cv,
                  'CV Scale', '', '', 
                  csfont, legend_label, lw, ylim=(0, 1.3))

SD_512_clockwork_goodput_slo = [112, 123, 166, 234, 256, 281, 378, 396]
SD_512_neustream_goodput_slo = [123, 212, 290, 376, 418, 453, 474, 483]
# P100_goodput = []
for idx in range(len(SD_512_clockwork_goodput_slo)):
    key = f"rate={1.25},cv={2.0}"
    duration = np.sum(trace_duration[key])
    SD_512_clockwork_goodput_slo[idx] /= duration
    SD_512_neustream_goodput_slo[idx] /= duration
    # P100_goodput.append(trace_request_count / duration)
plot_single_graph(axs[2, 1], slo, SD_512_clockwork_goodput_slo,
                  SD_512_neustream_goodput_slo,
                  'SLO Scale', '', '', 
                  csfont, legend_label, lw, ylim=(0, 1.2))

################################################################################################################
# below is data for DiT_S/2, img size = 512x512, RTX4090
################################################################################################################

# # Initialize parameters
# lw = 2.5
# csfont = {'fontname':"Times New Roman", 'size': 42}
# legend_fontsize = 24
# legend_label = ["Clockwork", "NeuStream"]

# # color mapping
# # Clockwork = orange
# # NeuStream = green

# # Data for the plots, img_size=256
# DiT_rate_img512 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# cv = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# slo = [1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5]

# f = open("DiT_S2_512_trace.json")
# trace_duration = json.load(f)

# trace_request_count = 500

# DiT_S2_512_clockwork_goodput_rate = [337, 294, 253, 250, 207, 203, 188, 179]
# DiT_S2_512_neustream_goodput_rate = [500, 499, 498, 497, 493, 480, 467, 462]
# # P100_goodput = []
# for idx in range(len(DiT_S2_512_clockwork_goodput_rate)):
#     key = f"rate={DiT_rate_img512[idx]},cv={2}"
#     duration = np.sum(trace_duration[key])
#     DiT_S2_512_clockwork_goodput_rate[idx] /= duration
#     DiT_S2_512_neustream_goodput_rate[idx] /= duration
#     # P100_goodput.append(trace_request_count / duration)
# plot_single_graph(axs[0, 2], DiT_rate_img512, DiT_S2_512_clockwork_goodput_rate,
#                   DiT_S2_512_neustream_goodput_rate, 
#                   'Rate Scale', '', 'DiT-S/2-img=512x512-RTX4090', 
#                   csfont, legend_label, lw, ylim=(0, 1.4))

# DiT_S2_512_clockwork_goodput_cv = [456, 376, 284, 263, 225, 184, 167, 154]
# DiT_S2_512_neustream_goodput_cv = [500, 500, 500, 498, 471, 442, 398, 369]
# P100_goodput = []

# # 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# # 所以用average rate对应的duration吧！
# duration = trace_request_count / 0.4
# for idx in range(len(DiT_S2_512_clockwork_goodput_cv)):
#     # key = f"rate={0.4},cv={cv[idx]}"
#     # duration = np.sum(trace_duration[key])
#     DiT_S2_512_clockwork_goodput_cv[idx] /= duration
#     DiT_S2_512_neustream_goodput_cv[idx] /= duration
#     # P100_goodput.append(trace_request_count / duration)
# plot_single_graph(axs[1, 2], cv, DiT_S2_512_clockwork_goodput_cv,
#                   DiT_S2_512_neustream_goodput_cv,
#                   'CV Scale', '', '', 
#                   csfont, legend_label, lw, ylim=(0, 1.2))

# DiT_S2_512_clockwork_goodput_slo = [141, 182, 213, 257, 294, 312, 323, 344]
# DiT_S2_512_neustream_goodput_slo = [182, 422, 484, 496, 498, 500, 500, 500]
# P100_goodput = []
# for idx in range(len(DiT_S2_512_clockwork_goodput_slo)):
#     key = f"rate={0.4},cv={2}"
#     duration = np.sum(trace_duration[key])
#     DiT_S2_512_clockwork_goodput_slo[idx] /= duration
#     DiT_S2_512_neustream_goodput_slo[idx] /= duration
#     # P100_goodput.append(trace_request_count / duration)
# plot_single_graph(axs[2, 2], slo, DiT_S2_512_clockwork_goodput_slo,
#                   DiT_S2_512_neustream_goodput_slo,
#                   'SLO Scale', '', '', 
#                   csfont, legend_label, lw, ylim=(0, 1.2))

################################################################################################################
# below is data for DiT_S/2, img size = 512x512, H100
################################################################################################################

# # Initialize parameters
# lw = 2.5
# csfont = {'fontname':"Times New Roman", 'size': 42}
# legend_fontsize = 24
# legend_label = ["Clockwork", "NeuStream"]

# # color mapping
# # Clockwork = orange
# # NeuStream = green

# # Data for the plots, img_size=256
# rate_img512 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# cv      =     [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# slo     =     [1.2, 1.5, 2, 2.5, 3, 3.5, 4]

# f = open("DiT_S2_512_H100_trace.json")
# trace_duration = json.load(f)

# trace_request_count = 500

# DiT_S2_512_clockwork_goodput_rate = [369, 344, 328, 300, 299, 281, 269, 259]
# DiT_S2_512_neustream_goodput_rate = [500, 499, 499, 496, 491, 485, 475, 468]
# # P100_goodput = []
# for idx in range(len(DiT_S2_512_clockwork_goodput_rate)):
#     key = f"rate={rate_img512[idx]},cv={2}"
#     duration = np.sum(trace_duration[key])
#     DiT_S2_512_clockwork_goodput_rate[idx] /= duration / 1.3333333333333333
#     DiT_S2_512_neustream_goodput_rate[idx] /= duration / 1.3333333333333333
#     # P100_goodput.append(trace_request_count / duration)
# plot_single_graph(axs[0, 3], rate_img512, DiT_S2_512_clockwork_goodput_rate,
#                   DiT_S2_512_neustream_goodput_rate,
#                   'Rate Scale', '', 'DiT-S/2-img=512x512-H100', 
#                   csfont, legend_label, lw, ylim=(0, 1.4))

# DiT_S2_512_clockwork_goodput_cv = [494, 438, 386, 330, 271, 244, 209, 198]
# DiT_S2_512_neustream_goodput_cv = [500, 500, 500, 499, 473, 433, 398, 358]
# # P100_goodput = []

# # 打印duration之后会发现，变化CV后的duration是波动的，而不是单调的。。。
# # 所以用average rate对应的duration吧！
# duration = trace_request_count / 0.7
# for idx in range(len(DiT_S2_512_clockwork_goodput_cv)):
#     # key = f"rate={0.7},cv={cv[idx]}"
#     # duration = np.sum(trace_duration[key])
#     DiT_S2_512_clockwork_goodput_cv[idx] /= duration / 1.3333333333333333
#     DiT_S2_512_neustream_goodput_cv[idx] /= duration / 1.3333333333333333
#     # P100_goodput.append(trace_request_count / duration)
# plot_single_graph(axs[1, 3], cv, DiT_S2_512_clockwork_goodput_cv,
#                   DiT_S2_512_neustream_goodput_cv,
#                   'CV Scale', '', '', 
#                   csfont, legend_label, lw, ylim=(0, 1.2))

# DiT_S2_512_clockwork_goodput_slo = [183, 207, 267, 302, 329, 354, 366]
# DiT_S2_512_neustream_goodput_slo = [346, 418, 472, 493, 499, 500, 500]
# # P100_goodput = []
# for idx in range(len(DiT_S2_512_clockwork_goodput_slo)):
#     key = f"rate={0.7},cv={2}"
#     duration = np.sum(trace_duration[key])
#     DiT_S2_512_clockwork_goodput_slo[idx] /= duration / 1.3333333333333333
#     DiT_S2_512_neustream_goodput_slo[idx] /= duration / 1.3333333333333333
#     # P100_goodput.append(trace_request_count / duration)
# plot_single_graph(axs[2, 3], slo, DiT_S2_512_clockwork_goodput_slo,
#                   DiT_S2_512_neustream_goodput_slo,
#                   'SLO Scale', '', '', 
#                   csfont, legend_label, lw, ylim=(0, 1.2))


################################################################################################################
# End data input
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
plt.savefig('goodput_high_workload.pdf', format='pdf')

# plt.show()
# Show the plot
