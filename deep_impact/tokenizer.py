import numpy as np

import torch

from transformers import BertTokenizerFast
from transformers.models.bert.tokenization_bert import BasicTokenizer

class QueryDocumentTokenizer:
    '''Class responsible to BERT-tokenize and BERT-encode a list of documents,
       with custom support for special tokens.'''

    def __init__(self, doc_maxlen: int):

        # Basic tokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
        self.basic = BasicTokenizer(do_lower_case=True, strip_accents=True)
        # WordPiece tokenizer working on already word-splitted text
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased', do_basic_tokenize=False)

        self.doc_maxlen = doc_maxlen

        self.cls_token    = self.tok.cls_token
        self.cls_token_id = self.tok.cls_token_id

        self.sep_token    = self.tok.sep_token
        self.sep_token_id = self.tok.sep_token_id


    def tensorize(self, batch_text):
        '''Convert a sequence of strings into a sequence of tuples.
           Each tuple contains the BERT's token ids and attention mask
           of the corresponding string.'''

        assert type(batch_text) in [list, tuple], (type(batch_text))

        # Basic tokenization
        batch_text = [self.basic.tokenize(text) for text in batch_text]
        # Bertization
        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=self.doc_maxlen, is_split_into_words=True)

        return obj


    def output_mask(self, q_batch, d_batch, d_tokenized_batch=None):
        '''For every query-document pairs, computes a boolean mask setting
           to True the indexes of the document token ids corresponding to first
           instance of every term in the query'''

        assert type(q_batch) in [list, tuple], (type(q_batch))
        assert type(d_batch) in [list, tuple], (type(d_batch))
        assert len(q_batch) == len(d_batch)

        # Basic tokenization
        q_batch = [list(set(self.basic.tokenize(text))) for text in q_batch]
        d_batch = [self.basic.tokenize(text) for text in d_batch]

        # Bertization
        if not d_tokenized_batch:
            d_tokenized_batch = self.tok(d_batch, padding='longest', truncation='longest_first',
                                         return_tensors='np', max_length=self.doc_maxlen, is_split_into_words=True)

        out_mask = np.zeros_like(d_tokenized_batch['input_ids'], dtype=np.bool)
        out_word2pos = []
        for q_terms, d_terms, d_encs, d_mask, out_mask_row in zip(q_batch, d_batch,
                                                                  d_tokenized_batch.encodings, d_tokenized_batch['attention_mask'],
                                                                  out_mask):
            qd_mask = [(t in q_terms) for t in d_terms] # elem is true if the corresponding doc terms is a query term
            word_ids_in_doc = [i for i, x in enumerate(qd_mask) if x] # convert previous mask in word_ids
            max_word_id_in_d_encs = d_encs.word_ids[d_mask.sum() - 2] # the maximum word id we can find in max_len tokens

            tok_ids = [d_encs.word_ids.index(word_id) for word_id in word_ids_in_doc if word_id < max_word_id_in_d_encs] # convert previous list in tok_ids
            out_mask_row[tok_ids] = True

            # We now compute the word to position maps, used at inference time, i.e., indexing

            out_word2pos.append(_compute_word2pos(d_terms, d_encs, word_ids_in_doc, max_word_id_in_d_encs))

        return torch.from_numpy(out_mask), out_word2pos

def _compute_word2pos(d_terms, d_encs, word_ids_in_doc, max_word_id_in_d_encs):

    word2tok = {}
    for word_id in word_ids_in_doc:
        if word_id < max_word_id_in_d_encs:
            word = d_terms[word_id]
            if word not in word2tok: # we keep only the first occurrence of a word
                word2tok[word] = d_encs.word_ids.index(word_id)
    return word2tok
