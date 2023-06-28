import torch
import torch.nn as nn

from random import sample, shuffle, randint

from transformers import *
import numpy as np



class DeepImpact(BertPreTrainedModel):
    def __init__(self, config):
        super(DeepImpact, self).__init__(config)
        self.bert = BertModel(config)
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )
        self.init_weights()
    
    def device(self, device):
        self.device = device


    def forward(self, encoded_list, masks):
        input_ids = torch.tensor([f.ids for f in encoded_list], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            [f.attention_mask for f in encoded_list], dtype=torch.long
        ).to(self.device)
        token_type_ids = torch.tensor([f.type_ids for f in encoded_list], dtype=torch.long).to(
            self.device
        )
        outputs = self.bert.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        doc_scores = self.impact_score_encoder(outputs)
        return doc_scores