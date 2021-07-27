import torch
import torch.nn as nn

from nltk.stem import PorterStemmer
from random import sample, shuffle, randint

from itertools import accumulate
from transformers import *
import re
from src.parameters import DEVICE
from src.utils2 import cleanQ, cleanD

stem = PorterStemmer().stem

MAX_LENGTH = 300


def unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


class MultiBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(MultiBERT, self).__init__(config)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def myforward(self, input_ids, attention_mask, token_type_ids):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.bert.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        return (sequence_output,)

    def encoder(self, hidden_states, attention_mask, head_mask):
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i == 7:
                break
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

        return (hidden_states,)


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
        term_indexes = [word_indexes.index(idx) for idx in match_indexes]

        a = [idx for i, idx in enumerate(match_indexes) if term_indexes[i] < MAX_LENGTH]
        b = [idx for idx in term_indexes if idx < MAX_LENGTH]

        return content, tokenized, a, b, len(word_indexes) + 2

    def forward(self, Q, D):
        bsize = len(Q)
        pairs = []
        X, pfx_sum, pfx_sumX = [], [], []
        total_size, total_sizeX, max_seq_length = 0, 0, 0

        doc_partials = []
        pre_pairs = []

        for q, d in zip(Q, D):
            tokens, tokenized, term_idxs, token_idxs, seq_length = self.tokenize(q, d)
            max_seq_length = max(max_seq_length, seq_length)

            pfx_sumX.append(total_sizeX)
            total_sizeX += len(term_idxs)

            tokens_split = tokens.split()

            doc_partials.append([(total_size + idx, tokens_split[i]) for idx, i in enumerate(term_idxs)])
            total_size += len(doc_partials[-1])
            pfx_sum.append(total_size)

            pre_pairs.append((tokenized, token_idxs))

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

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        y_score = self.classifier(pooled_output)
        y_score = torch.nn.functional.relu(y_score)

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

