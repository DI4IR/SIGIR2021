import os
import random
import torch
import torch.nn as nn

from argparse import ArgumentParser
from transformers import AdamW

from src.parameters import DEVICE, SAVED_CHECKPOINTS

from src.model import MultiBERT
from src.utils import print_message, save_checkpoint
import re
import datetime
import tqdm
import json
class TrainReader:
    def __init__(self, collection, triples, queries):
        #print_message("#> Training with the triples in", data_file, "...\n\n")
        expand_docs = [os.listdir(collection)][0]
        self.docs = {}
        self.queries = {}
        print("Reading Collection files")
        for fname in expand_docs:
            f = open(collection + "/" + fname,'r')
            print(collection + "/" + fname)
            for line in tqdm.tqdm(f):
                data = json.loads(line)
                pid = data["id"]
                content = data["contents"].strip()

                #pid, content = line.split("\t")
                self.docs[pid] = content.strip()
            f.close()

        print("Reading query files")
        f = open(queries,'r')
        for line in f:
            qid, q = line.split("\t")
            self.queries[qid] = q.strip()
        print(len(self.queries), len(self.docs))

        g = open(triples, 'r')
        self.train_data = []
        print("Creating training samples")
        for line in tqdm.tqdm(g):
            qid, pid1, pid2 = line.strip().split("\t")
            if qid not in self.queries or pid1 not in self.docs or pid2 not in self.docs:
                continue
            else:
                self.train_data.append([qid, pid1, pid2])
        self.i = 0

        #self.reader = open(data_file, mode='r', encoding="utf-8")
        

    def get_minibatch(self, bsize):
        ret = []
        for i in range(min(bsize, len(self.train_data) - self.i)):
            qid, pid1, pid2 = self.train_data[self.i + i]
            ret.append([self.queries[qid], self.docs[pid1], self.docs[pid2]])
        self.i += bsize
        return ret


def manage_checkpoints(colbert, optimizer, batch_idx, path="./"):
    if batch_idx % 2000 == 0:
        save_checkpoint("colbert-12layers-max300.dnn", 0, batch_idx, colbert, optimizer)

    if batch_idx in SAVED_CHECKPOINTS:
        save_checkpoint("colbert-12layers-max300-" + str(batch_idx) + ".dnn", 0, batch_idx, colbert, optimizer)


def train(args):
    colbert = MultiBERT.from_pretrained('bert-base-uncased')
    colbert = colbert.to(DEVICE)
    colbert.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(colbert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    reader = TrainReader(args.collections, args.triples, args.queries)
    train_loss = 0.0

    for batch_idx in range(args.maxsteps):
        Batch = reader.get_minibatch(args.bsize)
        Batch = sorted(Batch, key=lambda x: max(len(x[1]), len(x[2])))

        for B_idx in range(args.accumsteps):
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size: (B_idx+1) * size]
            Q, D1, D2 = zip(*B)

            colbert_out, _ = colbert(Q + Q, D1 + D2) #[Q1, Q2, ..., Qn, Q1, Q2, ..., Qn], [D1_1, ..., D1_1, D2_1, ..., D2_n]
            colbert_out= colbert_out.squeeze(1)

            colbert_out1, colbert_out2 = colbert_out[:len(Q)], colbert_out[len(Q):]

            out = torch.stack((colbert_out1, colbert_out2), dim=-1)

            positive_score, negative_score = round(colbert_out1.mean().item(), 2), round(colbert_out2.mean().item(), 2)
            print("#>>>   ", positive_score, negative_score, '\t\t|\t\t', positive_score - negative_score)
            loss = criterion(out, labels[:out.size(0)])
            loss = loss / args.accumsteps
            loss.backward()

            train_loss += loss.item()
            #print(loss.item())

        torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

        optimizer.step()
        optimizer.zero_grad()

        print_message(batch_idx, train_loss / (batch_idx+1))

        manage_checkpoints(colbert, optimizer, batch_idx+1, args.output_dir)
