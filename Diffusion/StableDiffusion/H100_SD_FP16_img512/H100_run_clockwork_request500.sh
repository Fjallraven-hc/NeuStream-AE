#!/bin/bash
# 在rtx4090/RTX4090 * 256/512下，有三种setting

# 此脚本跑rtx4090下的实验
mkdir "clockwork_request500_cv_log"
mkdir "clockwork_request500_slo_log"
mkdir "clockwork_request500_rate_log"

# 对于image_size=512, 默认值使用更大的slo_factor

# extra_time_for_vae&safety
# 保留4090上的设定
# image_size=256时, 设定为0.06s
# image_size=512时, 设定为0.1s

for device in "cuda:6"
do
    for rate_scale in "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2"
    do
        echo "Running with variable rate: $rate_scale"
        python SD_clockwork_infer_.py --log "clockwork_request500_rate_log" --image_size 512 --rate_scale $rate_scale --cv_scale 2.0 --slo_scale 3.0 --device $device
    done

    for cv_scale in "0.5" "1.0" "1.5" "2.0" "2.5" "3.0" "3.5" "4.0"
    do
        echo "Running with variable cv: $cv_scale"
        python SD_clockwork_infer_.py --log "clockwork_request500_cv_log" --image_size 512 --rate_scale 0.8 --cv_scale $cv_scale --slo_scale 3.0 --device $device
    done 

    for slo_scale in "1.2" "1.5" "2" "3" "4" "5" "6" "7"
    do
        echo "Running with variable slo: $slo_scale"
        python SD_clockwork_infer_.py --log "clockwork_request500_slo_log" --image_size 512 --rate_scale 0.8 --cv_scale 2.0 --slo_scale $slo_scale --device $device
    done
done