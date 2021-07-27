import torch
import torch.nn as nn

from nltk.stem import PorterStemmer
from random import sample, shuffle, randint
from transformers import *
import re
from itertools import accumulate
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

    def encoder(self, hidden_states, attention_mask, head_mask):
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i == 7:
                break
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

        return (hidden_states,)

    def myforward(self, input_ids, attention_mask, token_type_ids):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.bert.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        return (sequence_output,)

    def convert_example(self, d, max_seq_length):
        max_length = min(MAX_LENGTH, max_seq_length)
        inputs = self.tokenizer.encode_plus(d, add_special_tokens=True, max_length=max_length, truncation=True)

        padding_length = max_length - len(inputs["input_ids"])
        attention_mask = ([1] * len(inputs["input_ids"])) + ([0] * padding_length)
        input_ids = inputs["input_ids"] + ([0] * padding_length)
        token_type_ids = inputs["token_type_ids"] + ([0] * padding_length)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    
    def index(self, D, max_seq_length):
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

        pooled_output = torch.cat([hidden_state[i, list(map(lambda x: x[1], X[i]))] for i in range(bsize)])
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        y_score = self.classifier(pooled_output)
        y_score = torch.nn.functional.relu(y_score)
        y_score = y_score.squeeze().cpu().numpy().tolist()
        term_scores = [[(term, y_score[pos]) for term, _, pos in terms] for terms in X]

        return term_scores



