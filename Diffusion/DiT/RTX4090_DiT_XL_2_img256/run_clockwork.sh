mkdir clockwork_rate_log_request500
mkdir clockwork_cv_log_request500
mkdir clockwork_slo_log_request500

for rate in "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
do
   python DiT_XL_2_clockwork_infer.py --image_size 256 --rate_scale $rate --cv_scale 2 --slo_scale 3 --log_folder clockwork_rate_log_request500 --run_device cuda:2
done

for cv in "0.5" "1" "1.5" "2" "2.5" "3" "3.5" "4"
do
    python DiT_XL_2_clockwork_infer.py --image_size 256 --rate_scale 0.4 --cv_scale $cv --slo_scale 3 --log_folder clockwork_cv_log_request500 --run_device cuda:2
done

for slo in "1.2" "1.5" "2" "2.5" "3" "3.5" "4" "5"
do
    python DiT_XL_2_clockwork_infer.py --image_size 256 --rate_scale 0.4 --cv_scale 2 --slo_scale $slo --log_folder clockwork_slo_log_request500 --run_device cuda:2
done