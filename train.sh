python -m src.train \
    --collection /data/y247xie/00_data/MSMARCO/msmarco-passage-expanded/ \
    --triples /data/y247xie/00_data/MSMARCO/qidpidtriples.train.full.2.tsv \
    --queries /data/y247xie/00_data/MSMARCO/queries.train.tsv \
    --maxsteps 150000 \
    --bsize 32 \
    --accum 2 \
    --output_dir output.train_expand_retokenize_adjusted_len/ \
    --similarity cosine \
    --dim 128 \
    --query_maxlen 100 \
    --doc_maxlen 256


