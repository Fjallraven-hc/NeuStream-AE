# export PYTHONPATH=../:$PYTHONPATH
export MODEL_NAME="facebook/opt-66b"
export TP_SIZE=4
export NEU_DIR='./results'
export SAT_LEN=2048
export NUM_DEDUP=3
export CUDA_VISIBLE_DEVICES=0,1,2,3
# "meta-llama/Llama-2-13b-hf"
# facebook/opt-30b
# python prefill_time_prof.py --model $MODEL_NAME --tp $TP_SIZE --output_dir $NEU_DIR
# python decode_time_prof.py  --model $MODEL_NAME --tp $TP_SIZE --output_dir $NEU_DIR
python pd_simulate.py --model $MODEL_NAME --tp $TP_SIZE --output_dir $NEU_DIR --num_dedup $NUM_DEDUP --satuate_length $SAT_LEN
python prefill_time_true.py --model $MODEL_NAME --tp $TP_SIZE --output_dir $NEU_DIR --num_dedup $NUM_DEDUP