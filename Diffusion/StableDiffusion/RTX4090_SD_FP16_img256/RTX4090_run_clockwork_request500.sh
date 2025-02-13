#!/bin/bash
# 在rtx4090/RTX4090 * 256/512下，有三种setting

# 此脚本跑rtx4090下的实验
mkdir "request500_clockwork_cv_log"
mkdir "request500_clockwork_slo_log"
mkdir "request500_clockwork_rate_log"

# 对于image_size=512, 默认值使用更大的slo_factor

# extra_time_for_vae&safety
# 保留4090上的设定
# image_size=256时, 设定为0.06s
# image_size=512时, 设定为0.1s


for rate_scale in "0.75" "1.0" "1.25" "1.5" "1.75" "2.0" "2.25" "2.5"
do
    echo "Running with variable rate: $rate_scale"
    python SD_clockwork_infer_.py --log "request500_clockwork_rate_log" --image_size 256 --rate_scale $rate_scale --cv_scale 2.0 --slo_scale 3.0
done

for cv in "0.5" "1.0" "1.5" "2.0" "2.5" "3.0" "3.5" "4.0"
do
    echo "Running with variable cv: $cv"
    python SD_clockwork_infer_.py --log "request500_clockwork_cv_log" --image_size 256 --rate_scale 1.25 --cv_scale $cv --slo_scale 3.0
done 

for slo_scale in "1.2" "1.5" "2" "3" "4" "5" "6" "7"
do
    echo "Running with variable slo: $slo_scale"
    python SD_clockwork_infer_.py --log "request500_clockwork_slo_log" --image_size 256 --rate_scale 1.25 --cv_scale 2.0 --slo_scale $slo_scale
done
