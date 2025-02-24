# export PYTHONPATH=../:$PYTHONPATH
export MODEL_NAME="facebook/opt-13b"
export TP_SIZE=1
export NEU_DIR='./results'
export SAT_LEN=1024
export NUM_DEDUP=3
python pd_simulate.py --model $MODEL_NAME --tp $TP_SIZE --output_dir $NEU_DIR --num_dedup $NUM_DEDUP --satuate_length $SAT_LEN
python prefill_time_true.py --model $MODEL_NAME --tp $TP_SIZE --output_dir $NEU_DIR --num_dedup $NUM_DEDUP