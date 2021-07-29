import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from argparse import ArgumentParser

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
load_checkpoint("../../DeepImpact/official/colbert-12layers-100000.dnn", net)
net.eval()


def tok(d):
    cont = d
    d = cleanD(d, join=False)
    content = ' '.join(d)
    tokenized_content = net.tokenizer.tokenize(content)

    terms = list(set([(t, d.index(t)) for t in d]))  # Quadratic!
    word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
    terms = [(t, word_indexes.index(idx)) for t, idx in terms]
    terms = [(t, idx) for (t, idx) in terms if idx < MAX_LENGTH]

    return tokenized_content, terms, cont

def quantize(value, scale):
    return int(ceil(value * scale))


def process_batch(g, super_batch):
    print_message("Start process_batch()", "")
    scale = (1 << 8) / 21.0

    with torch.no_grad():
        if not len(super_batch):
            return None
        super_batch = list(p.map(tok, super_batch))
        #super_batch = [tok(x) for x in super_batch]

        sorted_super_batch = sorted([(v, idx) for idx, v in enumerate(super_batch)], key=lambda x: len(x[0][0]))
        super_batch = [v for v, _ in sorted_super_batch]
        super_batch_indices = [idx for _, idx in sorted_super_batch]

        print_message("Done sorting", "")

        every_term_score = []
        contents = []

        for batch_idx in range(ceil(len(super_batch) / MB_SIZE)):
            D = super_batch[batch_idx * MB_SIZE: (batch_idx + 1) * MB_SIZE]
            IDXs = super_batch_indices[batch_idx * MB_SIZE: (batch_idx + 1) * MB_SIZE]
            all_term_scores,cont  = net.index(D, len(D[-1][0])+2)
            every_term_score += zip(IDXs, all_term_scores)
            contents += zip(IDXs, cont)

        every_term_score = sorted(every_term_score)
        contents = sorted(contents)

        lines = []
        #for _, term_scores in every_term_score:
        #    term_scores = ', '.join([term + ": " + str(int(quantize(score, scale))) for term, score in term_scores])
        #    lines.append(term_scores)
        for idx, term_scores in enumerate(every_term_score):
            _, ts = term_scores
            data = {
                    "id":idx,
                    "contents": contents[idx][1],
                    "vector":{}
                    }
            for t, s in ts:
                data["vector"][t] = quantize(s, scale)
            g.write(json.dumps(data) + "\n")

    g.flush()

if __name__ == "__main__":
    parser = ArgumentParser(description='Eval ColBERT with <query, positive passage, negative passage> triples.')
    
    #parser.add_argument('--bsize', dest='bsize', default=32, type=int)
    #parser.add_argument('--triples', dest='triples', default='triples.train.small.tsv')
    #parser.add_argument('--output_dir', dest='output_dir', default='outputs.train/')
    #parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])

    parser.add_argument('--collection', default="./baseline_test", type=str)# collection file: tsv, docid \t doc
    parser.add_argument('--output_path', default="./collections/", type=str)
    #parser.add_argument('--output_name', default="/index-July13%d.txt", type=str)
    #parser.add_argument('--query_path', default="./collection-dT5-newterms_unique.tsv", type=str)
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--ckpt', default='./colbert-12layers-max300-32000.dnn',type=str)

    args = parser.parse_args()
    args.input_arguments = args

    print_message("#> Loading model checkpoint.")
    net = MultiBERT.from_pretrained('bert-base-uncased')
    DEVICE = "cuda"
    net = net.to(DEVICE)
    load_checkpoint(args.ckpt, net)
    net.eval()


    p = Pool(16)
    start_time = time()
    #COLLECTION = "./baseline_test"
    g = open(args.output_path+ "doc0.json", 'w')
    #f = open(args.query_path)
    text_id = 0
    #expand_docs = [os.listdir(args.query_path)][:1]


    #for fname in expand_docs:
    #    f = open(fname, 'r')
    with open(args.collection, 'r') as f:

        for idx, passage in enumerate(f):
            #data = json.loads(passage)
            #id_ = data["id"]
            #contents = data["contents"]

            if idx % (50*1024) == 0:
                if idx > 0:
                    process_batch(g, super_batch)
                throughput = round(idx / (time() - start_time), 1)
                print_message("Processed", str(idx), "passages so far [rate:", str(throughput), "passages per second]")
                super_batch = []

            passage = passage.strip()
            pid, passage = passage.split('\t')
            super_batch.append(passage)

            if idx % 1000000 == 999999:
                g.close()
                text_id += 1
                g = open(args.output_path + "/doc{}.json",format(text_id), "w")

process_batch(g, super_batch)
g.close()
f.close()

