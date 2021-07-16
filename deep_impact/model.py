#pylint: disable=W0223

from torch import nn

from transformers import BertPreTrainedModel, BertModel

from deep_impact import parameters
from deep_impact.tokenizer import QueryDocumentTokenizer


class DeepImpactModel(BertPreTrainedModel):

    def __init__(self, config):

        # TODO: place the args.max_doc_len somewhere
        super().__init__(config)

        self.tokenizer = QueryDocumentTokenizer(doc_maxlen=parameters.MAX_LENGTH)

        self.bert = BertModel(config)

        self.impacter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )

        self.init_weights()

    def forward(self, q_batch, d_batch):
        # Input:
        #   q_batch = ( q1,  q2,  q3,  q1,  q2,  q3)
        #   d_batch = (d+1, d+2, d+3, d-1, d-2, d-3)
        # Output:
        #   y_score = (s+1, s+2, s+3, s-1, s-2, s-3)

        d_tokenized_batch = self.tokenizer.tensorize(d_batch)
        input_ids      = d_tokenized_batch['input_ids'].to(parameters.DEVICE)
        attention_mask = d_tokenized_batch['attention_mask'].to(parameters.DEVICE)

        bert_outputs = self.bert.forward(input_ids, attention_mask)[0] # batch_size x doc size x emb size

        output_masks, output_terms = self.tokenizer.output_mask(q_batch, d_batch)#, d_tokenized_batch) # batch_size x doc size x emb size
        output_masks = output_masks.to(parameters.DEVICE)

        pooled_output = output_masks.unsqueeze(-1) * bert_outputs # batch_size x doc size x emb size

        y_scores = self.impacter(pooled_output)

        # We now compute term scores, useful at inference time, i.e., indexing
        term_scores = _compute_term_scores(output_terms, y_scores)

        return y_scores.squeeze(-1).sum(dim=1), term_scores # We return a loss value for every document and the term scores


def _compute_term_scores(output_terms, y_scores ):

    term_scores = []
    for word2pos, doc_score in zip(output_terms, y_scores):
        word2score = {}
        for word, pos in word2pos.items():
            word2score[word] = doc_score[pos].item()
        term_scores.append(word2score)
    return term_scores
