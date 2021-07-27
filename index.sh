export CKPT=64000
export CUDA_LAUNCH_BLOCKING=1
python -m src.index \
    --output_name ckpt_${CKPT} \
    --collection indexes_new/collection.train_expand_retokenize_adjusted_len/ckpt_${CKPT} \
    --query_path /data/y247xie/00_data/MSMARCO/msmarco-passage-expanded/  \
    --ckpt ckpts/output.train_expand_retokenize_adjusted_len/colbert-12layers-max300-${CKPT}.dnn

#    /data/y247xie/00_data/MSMARCO/msmarco-passage-expanded/docsxx.json \

