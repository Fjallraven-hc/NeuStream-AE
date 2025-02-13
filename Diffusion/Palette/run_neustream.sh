mkdir neustream_rate_log_request500
mkdir neustream_cv_log_request500
mkdir neustream_slo_log_request500

for step_delta in "0.95" # "1" # "0.9"
do
    for rate in "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
    do
       python Palette_inpainting_places2_our_system.py --image_size 256 --rate_scale $rate --cv_scale 2 --slo_scale 3 --log_folder neustream_rate_log_request500 --run_device cuda:4 --profile_device RTX4090 --step_delta $step_delta
    done

    for cv in "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4"
    do
        python Palette_inpainting_places2_our_system.py --image_size 256 --rate_scale 0.4 --cv_scale $cv --slo_scale 3 --log_folder neustream_cv_log_request500 --run_device cuda:4 --profile_device RTX4090 --step_delta $step_delta
    done

    for slo in "1.2" "1.5" "2" "2.5" "3" "3.5" "4" "5"
    do
        python Palette_inpainting_places2_our_system.py --image_size 256 --rate_scale 0.4 --cv_scale 2 --slo_scale $slo --log_folder neustream_slo_log_request500 --run_device cuda:4 --profile_device RTX4090 --step_delta $step_delta
    done
done