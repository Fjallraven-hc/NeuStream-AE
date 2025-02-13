mkdir neustream_rate_log_request500
mkdir neustream_cv_log_request500
mkdir neustream_slo_log_request500

for step_delta in "1" # "1" # "0.9"
do # "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4"
    for rate in "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
    do
        python DiT_XL_2_our_system.py --image_size 256 --rate_scale $rate --cv_scale 2 --slo_scale 3 --log_folder neustream_rate_log_request500 --run_device cuda:1 --extra_vae_safety_time 0.1 --profile_device RTX4090 --step_delta $step_delta
    done

    for cv in "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4"
    do
        python DiT_XL_2_our_system.py --image_size 256 --rate_scale 0.4 --cv_scale $cv --slo_scale 3 --log_folder neustream_cv_log_request500 --run_device cuda:1 --extra_vae_safety_time 0.1 --profile_device RTX4090 --step_delta $step_delta
    done

    for slo in "1.2" "1.5" "2" "2.5" "3" "3.5" "4" "5"
    do
        python DiT_XL_2_our_system.py --image_size 256 --rate_scale 0.4 --cv_scale 2 --slo_scale $slo --log_folder neustream_slo_log_request500 --run_device cuda:1 --extra_vae_safety_time 0.1 --profile_device RTX4090 --step_delta $step_delta
    done
done