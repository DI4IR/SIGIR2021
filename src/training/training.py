
from src.training.msmarco_batcher import MSMarcoBatcher
import torch.nn as nn
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.optim as optim
from transformers import set_seed

from torch.utils.data import DataLoader

import torch
from src.training.model_checkpoint import ModelCheckpoint
import os
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from src.model.deep_impact import DeepImpact
from tokenizers import Tokenizer
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


tokenizer = Tokenizer.from_pretrained('bert-base-uncased')
basic_tokenizer = tokenizer.pre_tokenizer
normalizer = tokenizer.normalizer



def load_checkpoint(model, optimizer, checkpoint):
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



def collate(q, d, max_length):
    q = normalizer.normalize_str(q)
    results = list(set(basic_tokenizer.pre_tokenize_str(q.lower())))
    if len(results) > 0:
        query_tokens, _ = zip(*results)
    else:
        query_tokens = []

    d = normalizer.normalize_str(d)
    results = basic_tokenizer.pre_tokenize_str(d.lower())
    if len(results) > 0:
        doc_tokens, _ = zip(*results)
    else:
        doc_tokens = []
    doc_tokens = doc_tokens[:max_length]

    encoded = tokenizer.encode(" ".join(doc_tokens))
    tokenized = encoded.tokens[1:]  # ignore CLS token

    word_indexes = list(
        accumulate([-1] + tokenized, lambda a, b: a + int(not b.startswith("##")))
    )
    match_indexes = list(set([doc_tokens.index(t) for t in query_tokens if t in doc_tokens]))
    term_indexes = [word_indexes.index(idx) for idx in match_indexes if idx in word_indexes and word_indexes.index(idx) < max_length]
    mask = np.zeros(max_length, dtype=bool)
    mask[term_indexes] = True
    return encoded, torch.from_numpy(mask)



def train(args):
    set_seed(12345)
    torch.cuda.set_device(args.rank)
    device = torch.device("cuda:{}".format(args.rank))
    torch.distributed.init_process_group(backend="nccl")


    tokenizer.enable_truncation(args.max_length)
    tokenizer.enable_padding(length=args.max_length)

    def collate_fn(examples):
        E = []
        M = []
        L = []
        for i in range(0, len(examples)):
            query = examples[i][0]
            document = examples[i][1]
            encoded, mask = collate(query, document, args.max_length)
            E.append(encoded)
            M.append(mask)
            L.append(1)

            document = examples[i][2]
            encoded, mask = collate(query, document, args.max_length)
            E.append(encoded)
            M.append(mask)
            L.append(0)
        return [E, M, L]


    nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
    nranks = max(1, nranks)
    assert args.batch_size % nranks == 0, (args.batch_size, nranks)
    args.batch_size = args.batch_size // nranks

    dataset = MSMarcoBatcher(args)
    train_sampler = DistributedSampler(dataset=dataset)

    train_dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                num_workers=32,
                sampler=train_sampler,
                shuffle=False,
                drop_last=True,
    )

    scaler = amp.GradScaler()
    model = DeepImpact.from_pretrained("bert-base-uncased")
    model.device(device)
    model = model.to(device)

    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)

    if args.checkpoint:
        load_checkpoint(model, optimizer, args.checkpoint)


    model.train()
    criterion = nn.CrossEntropyLoss()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank],
                                                      output_device=args.rank,
                                                      find_unused_parameters=True)



    checkpoint_callback = ModelCheckpoint(
        model, optimizer, "./checkpoints/", "deepimpact", 2000)
    
    while True:
        with tqdm(total=len(train_dataloader)) as pbar:
            train_loss = 0
            for step, batch in enumerate(train_dataloader):
                with torch.cuda.amp.autocast():
                    E, M, L = batch
                    labels = torch.tensor(L, dtype=torch.float, device=device)
                    labels = labels.view(args.batch_size, -1)
                    doc_scores = model(E, M)
                    mask = torch.stack(M, dim=0).to(device).unsqueeze(-1)
                    outputs = (mask * doc_scores).sum(dim=1)
                    outputs = outputs.squeeze()
                    outputs = outputs.view(args.batch_size, -1)


                    loss = criterion(outputs, labels)

                    
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                train_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if args.rank == 0:
                    pbar.update(1)
                    pbar.set_description("Avg train loss={:.2f}, Examples seen={}".format(
                        train_loss/(step+1)*100, step * args.batch_size * nranks))
                    checkpoint_callback()
