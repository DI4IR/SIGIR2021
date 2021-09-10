import torch
import torch.nn as nn

from random import sample, shuffle, randint

from itertools import accumulate
from transformers import *
import re
from src.parameters import DEVICE
from src.utils2 import cleanQ, cleanD

MAX_LENGTH = 300


class MultiBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(MultiBERT, self).__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )
        self.init_weights()

    def convert_example(self, d, max_seq_length):
        max_length = min(MAX_LENGTH, max_seq_length)
        inputs = self.tokenizer.encode_plus(d, add_special_tokens=True, max_length=max_length, truncation=True)

        padding_length = max_length - len(inputs["input_ids"])
        attention_mask = ([1] * len(inputs["input_ids"])) + ([0] * padding_length)
        input_ids = inputs["input_ids"] + ([0] * padding_length)
        token_type_ids = inputs["token_type_ids"] + ([0] * padding_length)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    def tokenize(self, q, d):
        query_tokens = list(set(cleanQ(q).strip().split()))  # [:10]

        content = cleanD(d).strip()
        doc_tokens = content.split()

        # NOTE: The following line accounts for CLS!
        tokenized = self.tokenizer.tokenize(content)
        word_indexes = list(accumulate([-1] + tokenized, lambda a, b: a + int(not b.startswith('##'))))
        match_indexes = list(set([doc_tokens.index(t) for t in query_tokens if t in doc_tokens]))
        match_queries = list(set([query_tokens.index(t) for t in query_tokens if (t in doc_tokens and word_indexes.index(doc_tokens.index(t)) < MAX_LENGTH)]))
        term_indexes = [word_indexes.index(idx) for idx in match_indexes]

        a = [idx for i, idx in enumerate(match_indexes) if term_indexes[i] < MAX_LENGTH]
        b = [idx for idx in term_indexes if idx < MAX_LENGTH]

        return content, tokenized, a, b, len(word_indexes) + 2, match_queries

    def tok(self, D):
        T = []
        for d in D:
            content = cleanD(d).strip()
            doc_tokens = content.split()
            tokenized_content = self.tokenizer.tokenize(content)

            terms = list(set([(t, doc_tokens.index(t)) for t in doc_tokens]))  # Quadratic!
            word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
            terms = [(t, word_indexes.index(idx)) for t, idx in terms]
            terms = [(t, idx) for (t, idx) in terms if idx < MAX_LENGTH]
        
            T.append((tokenized_content, terms))
        return T


    def forward(self, Q, D):
        bsize = len(Q)
        pairs = []
        X, pfx_sum, pfx_sumX = [], [], []
        total_size, total_sizeX, max_seq_length = 0, 0, 0

        doc_partials = []
        pre_pairs = []
        Y = []
        for q, d in zip(Q, D):
            tokens, tokenized, term_idxs, token_idxs, seq_length, match_queries = self.tokenize(q, d)
            max_seq_length = max(max_seq_length, seq_length)

            pfx_sumX.append(total_sizeX)
            total_sizeX += len(term_idxs)

            tokens_split = tokens.split()

            doc_partials.append([(total_size + idx, tokens_split[i]) for idx, i in enumerate(term_idxs)])
            total_size += len(doc_partials[-1])
            pfx_sum.append(total_size)

            pre_pairs.append((tokenized, token_idxs))
            Y.append(match_queries)
        if max_seq_length % 10 == 0:
            print("#>>>   max_seq_length = ", max_seq_length)

        for tokenized, token_idxs in pre_pairs:
            pairs.append(self.convert_example(tokenized, max_seq_length))
            X.append(token_idxs)

        input_ids = torch.tensor([f['input_ids'] for f in pairs], dtype=torch.long).to(DEVICE)
        attention_mask = torch.tensor([f['attention_mask'] for f in pairs], dtype=torch.long).to(DEVICE)
        token_type_ids = torch.tensor([f['token_type_ids'] for f in pairs], dtype=torch.long).to(DEVICE)

        outputs = self.bert.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs[0]
        def one(i):
            if len(X[i]) > 0:
                l = [hidden_state[i, j] for j in X[i]]  # + [mismatch_scores[i, j] for j in all_mismatches[i]]
                return torch.stack(l)
            return torch.tensor([]).to(DEVICE)

        pooled_output = torch.cat([one(i) for i in range(bsize)])

        bsize = len(pooled_output)

        if bsize == 0:
            term_scores = []
            for doc in doc_partials:
                term_scores.append([])
                for (idx, term) in doc:
                    term_scores[-1].append((term, 0.0))

            return torch.tensor([[0.0]] * len(Q)).to(DEVICE), term_scores

        y_score = self.impact_score_encoder(pooled_output)

        q_score = self.index(self.tok(Q), max_seq_length, Y)
            
        y_score = y_score * q_score

        x = torch.arange(bsize).expand(len(pfx_sum), bsize) < torch.tensor(pfx_sum).unsqueeze(1)
        y = torch.arange(bsize).expand(len(pfx_sum), bsize) >= torch.tensor([0] + pfx_sum[:-1]).unsqueeze(1)
        mask = (x & y).to(DEVICE)
        y_scorex = list(y_score.cpu())
        term_scores = []
        for doc in doc_partials:
            term_scores.append([])
            for (idx, term) in doc:
                term_scores[-1].append((term, y_scorex[idx]))

        return (mask.type(torch.float32) @ y_score), term_scores #, ordered_terms #, num_exceeding_fifth

    def index(self, D, max_seq_length, Y):
        if max_seq_length % 10 == 0:
            print("#>>>   max_seq_length = ", max_seq_length)

        bsize = len(D)
        offset = 0
        pairs, X = [], []

        for tokenized_content, terms in D:
            terms = [(t, idx, offset + pos) for pos, (t, idx) in enumerate(terms)]
            offset += len(terms)
            pairs.append(self.convert_example(tokenized_content, max_seq_length))
            X.append(terms)

        input_ids = torch.tensor([f['input_ids'] for f in pairs], dtype=torch.long).to(DEVICE)
        attention_mask = torch.tensor([f['attention_mask'] for f in pairs], dtype=torch.long).to(DEVICE)
        token_type_ids = torch.tensor([f['token_type_ids'] for f in pairs], dtype=torch.long).to(DEVICE)

        outputs = self.bert.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_state = outputs[0]


        def one(i):
            if len(Y[i]) > 0:
                l = [hidden_state[i, j] for j in Y[i]]  # + [mismatch_scores[i, j] for j in all_mismatches[i]]
                return torch.stack(l)
            return torch.tensor([]).to(DEVICE)

        pooled_output = torch.cat([one(i) for i in range(bsize)])


#        pooled_output = torch.cat([hidden_state[i, list(map(lambda x: x[1], X[i]))] for i in range(bsize)])

        y_score = self.impact_score_encoder(pooled_output)

        return y_score

