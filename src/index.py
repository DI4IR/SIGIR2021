import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from math import ceil
from src.model import *
from multiprocessing import Pool
from src.evaluation.loaders import load_checkpoint

MB_SIZE = 1024

def print_message(*s):
    s = ' '.join(map(str, s))
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


print_message("#> Loading model checkpoint.")
net = MultiBERT.from_pretrained('bert-base-uncased')
net = net.to(DEVICE)
load_checkpoint("/scratch/am8949/MultiBERT/colbert-12layers-100000.dnn", net)
net.eval()






def tok(d):
    d = cleanD(d, join=False)
    content = ' '.join(d)
    tokenized_content = net.tokenizer.tokenize(content)

    terms = list(set([(t, d.index(t)) for t in d]))  # Quadratic!
    word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
    terms = [(t, word_indexes.index(idx)) for t, idx in terms]
    terms = [(t, idx) for (t, idx) in terms if idx < MAX_LENGTH]

    return tokenized_content, terms



def process_batch(g, super_batch):
    print_message("Start process_batch()", "")

    with torch.no_grad():
        super_batch = list(p.map(tok, super_batch))

        sorted_super_batch = sorted([(v, idx) for idx, v in enumerate(super_batch)], key=lambda x: len(x[0][0]))
        super_batch = [v for v, _ in sorted_super_batch]
        super_batch_indices = [idx for _, idx in sorted_super_batch]

        print_message("Done sorting", "")

        every_term_score = []

        for batch_idx in range(ceil(len(super_batch) / MB_SIZE)):
            D = super_batch[batch_idx * MB_SIZE: (batch_idx + 1) * MB_SIZE]
            IDXs = super_batch_indices[batch_idx * MB_SIZE: (batch_idx + 1) * MB_SIZE]
            all_term_scores = net.index(D, len(D[-1][0])+2)
            every_term_score += zip(IDXs, all_term_scores)

        every_term_score = sorted(every_term_score)

        lines = []
        for _, term_scores in every_term_score:
            term_scores = ', '.join([term + ": " + str(round(score, 3)) for term, score in term_scores])
            lines.append(term_scores)

    g.write('\n'.join(lines) + "\n")
    g.flush()


p = Pool(16)
start_time = time()

COLLECTION = "/scratch/am8949"
with open(COLLECTION + '/queries.dev.test.txt', 'w') as g:
    with open(COLLECTION + '/queries.dev.small.tsv') as f:
        for idx, passage in enumerate(f):
            if idx % (50*1024) == 0:
                if idx > 0:
                    process_batch(g, super_batch)
                throughput = round(idx / (time() - start_time), 1)
                print_message("Processed", str(idx), "passages so far [rate:", str(throughput), "passages per second]")
                super_batch = []

            passage = passage.strip()
            pid, passage = passage.split('\t')
            super_batch.append(passage)

            #assert int(pid) == idx

        process_batch(g, super_batch)

