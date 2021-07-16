from more_itertools import chunked

import torch
from torch import nn
from torch.optim import Optimizer # typing import

from transformers import AdamW
from transformers import BertModel # typing import

from deep_impact.parameters import DEVICE, SAVED_CHECKPOINTS, MODEL_NAME_PREFIX
from deep_impact.utils import print_message, save_checkpoint
from deep_impact.model import DeepImpactModel


def read_minibatch_iter(data_file: str, bsize: int) -> list[(str, str, str)]:

    print_message("#> Training with the triples in", data_file, "...")

    with open(data_file, mode="r", encoding="utf-8") as reader:
        batch = []
        for line in reader:
            batch.append(line.split('\t'))
            if len(batch) == bsize:
                batch = sorted(batch, key=lambda x: max(len(x[1]), len(x[2])))
                yield batch
                batch = []


def manage_checkpoints(model: BertModel, optimizer: Optimizer, batch_idx: int) -> None:

    if batch_idx % 2000 == 0:
        # save_checkpoint("colbert-12layers-max300.dnn", 0, batch_idx, model, optimizer)
        save_checkpoint(MODEL_NAME_PREFIX + ".dnn", 0, batch_idx, model, optimizer)

    if batch_idx in SAVED_CHECKPOINTS:
        # save_checkpoint("colbert-12layers-max300-" + str(batch_idx) + ".dnn",
        #                 0, batch_idx, model, optimizer)
        save_checkpoint(MODEL_NAME_PREFIX + f"-{batch_idx}.dnn", 0, batch_idx, model, optimizer)


def train(bsize, triples, maxsteps, lr, accumsteps):

    di_model = DeepImpactModel.from_pretrained('bert-base-uncased')
    di_model = di_model.to(DEVICE)
    di_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(di_model.parameters(), lr=lr, eps=1e-8)

    optimizer.zero_grad()
    labels = torch.zeros(bsize, dtype=torch.long, device=DEVICE)

    train_loss = 0.0

    for epoch, batch in enumerate(read_minibatch_iter(triples, bsize)):

        for chunk in chunked(batch, accumsteps):
            query, doc_pos, doc_neg = zip(*chunk)

#            print("query:", query)
#            print("doc_pos:", doc_pos)
#            print("doc_neg:", doc_neg)
#            print("chunk:", chunk)

            di_out, _ = di_model.forward(query + query, doc_pos + doc_neg)

            di_out_pos, di_out_neg = di_out[:len(query)], di_out[len(query):]
            out = torch.stack((di_out_pos, di_out_neg), dim=-1)

            loss = criterion(out, labels[:out.size(0)])
            loss = loss / accumsteps
            loss.backward()

            train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(di_model.parameters(), 2.0)

        optimizer.step()
        optimizer.zero_grad()

        print_message(epoch, train_loss / (epoch + 1))
        manage_checkpoints(di_model, optimizer, epoch + 1)

        if epoch >= maxsteps:
            break
