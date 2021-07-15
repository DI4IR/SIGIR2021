python -m src.train \
    --triples ./triples.train.small.tsv \
    --maxsteps 100000 \
    --bsize 32 \
    --accum 2 \
    --output_dir output.train/ \
    --similarity cosine \
    --dim 128 \
    --query_maxlen 32 \
    --doc_maxlen 180


