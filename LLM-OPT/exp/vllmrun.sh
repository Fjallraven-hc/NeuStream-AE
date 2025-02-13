# export $MODEL="facebook/opt-30b"
# export $TP_SIZE=1
# export $NEU_LOG_DIR="./logs/vllm_rate"
# export CUDA_VISIBLE_DEVICES=1
# python poisson_experiment.py --gamma --vllm # --model $MODEL --tp_size $TP_SIZE --neu_dir $NEU_DIR
# DIR="./paper_results/refine/vllm_opt66b_tp4_rate1-5_cv1_slo1.5+1.5_lmchat_1000req_seedmap"
DIR="./paper_results/refine/vllm_final"
CUDA_VISIBLE_DEVICES=0,1,2,3 NEU_LOG_DIR=$DIR python poisson_experiments.py --gamma --vllm  --rate 3 --cv 1 --pslo 1.5
# for rate in 5
# do
#     CUDA_VISIBLE_DEVICES=6,7 NEU_LOG_DIR=$DIR python poisson_experiments.py --gamma -rid 0 -rws 2 --rate $rate --vllm
#     CUDA_VISIBLE_DEVICES=6,7 NEU_LOG_DIR=$DIR python poisson_experiments.py --gamma -rid 1 -rws 2 --rate $rate --vllm
#     # CUDA_VISIBLE_DEVICES=6,7 NEU_LOG_DIR=$DIR python poisson_experiments.py --gamma -rid 0 -rws 3 --rate $rate --vllm
#     # CUDA_VISIBLE_DEVICES=6,7 NEU_LOG_DIR=$DIR python poisson_experiments.py --gamma -rid 1 -rws 3 --rate $rate --vllm
#     # CUDA_VISIBLE_DEVICES=6,7 NEU_LOG_DIR=$DIR python poisson_experiments.py --gamma -rid 2 -rws 3 --rate $rate --vllm
# done