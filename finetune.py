#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import module
import json
import os
import nltk
import torch
import json
import argparse
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import collections
import logging
import random
import time
import numpy as np
import math
import evaluate
import copy

from utils.nn import LSTM, Linear
from time import localtime, strftime

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# In[2]:


#argument parser
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model checkpoints and predictions will be written.")

## Other parameters
parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
parser.add_argument("--predict_file", default=None, type=str,
                    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument("--max_seq_length", default=450, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
parser.add_argument("--predict_batch_size", default=64, type=int, help="Total batch size for predictions.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                         "of training.")
parser.add_argument("--n_best_size", default=1, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json "
                         "output file.")
parser.add_argument("--max_answer_length", default=30, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--version_2_with_negative',
                    action='store_true',
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold',
                    type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")

parser.add_argument('--char-dim', default=8, type=int)
parser.add_argument('--char-channel-width', default=5, type=int)
parser.add_argument('--char-channel-size', default=100, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--epoch', default=5, type=int)
parser.add_argument('--exp-decay-rate', default=0.999, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--hidden-size', default=100, type=int)
parser.add_argument('--learning-rate', default=5e-5, type=float)
parser.add_argument('--print-freq', default=250, type=int)
parser.add_argument('--word-dim', default=100, type=int)
parser.add_argument('--id', default=0, type=int)
parser.add_argument('--bert-layer-size', default=768, type=int)
parser.add_argument('--saved_model', default=None, type=str)

args = parser.parse_args()


# In[3]:


class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        #self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        #nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        #self.char_conv = nn.Conv1d(args.char_dim, args.char_channel_size, args.char_channel_width)

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        #self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        # assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        #assert self.args.word_dim == 2 * self.args.hidden_size
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(2 * args.hidden_size, 2 * args.hidden_size),
                                  nn.ReLU(inplace=True)))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(2 * args.hidden_size, 2 * args.hidden_size),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.bert_layer_size * 4,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_q = Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1)
#         self.att_weight_self1 = Linear(args.hidden_size * 2, 1)
#         self.att_weight_self2 = Linear(args.hidden_size * 2, 1)
#         self.att_weight_self3 = Linear(args.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=args.hidden_size * 8,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        self.modeling_LSTM2 = LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        # 6. Output Layer
        self.p1_weight = Linear(args.hidden_size * 10, 1, dropout=args.dropout)
        self.p2_weight = Linear(args.hidden_size * 10, 1, dropout=args.dropout)

        self.output_LSTM = LSTM(input_size=args.hidden_size * 2,
                                hidden_size=args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout, inplace=True)

    def forward(self,c_word, q_word, c_lens, q_lens,c_mask,q_mask):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len,char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(2))
            # (batch * seq_len, char_channel_size,conv_len)
            x = self.char_conv(x)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q, c_mask, q_mask):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)


            s_cq = self.att_weight_c(c).expand(-1,-1,q_len) +                    self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) +                    self.att_weight_cq((c.unsqueeze(2))*(q.unsqueeze(1))).squeeze(-1)
            mask = (1.0 - c_mask.unsqueeze(2) * q_mask.unsqueeze(1)) * -10000.0
            
            assert s_cq.size()==mask.size()
            
            s_cq = s_cq + mask

            # (batch, c_len, q_len)
            a = F.softmax(s_cq, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s_cq, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight(torch.cat((g,m),dim=-1))).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            m2_padded = F.pad(m2, (0,0,0,g.size(1) - m2.size(1)))
            # (batch, c_len)
            p2 = (self.p2_weight(torch.cat((g,m2_padded),dim=-1))).squeeze()

            return p1, p2

        # 1. Character Embedding Layer

        c_maxlen = c_word.size()[1]
        q_maxlen = q_word.size()[1]


#         # Highway network
#         c_cat = torch.cat([c_char, c_word], dim=-1)
#         q_cat = torch.cat([q_char, q_word], dim=-1)
#         c = highway_network(c_cat)
#         q = highway_network(q_cat)

#         # 3. Contextual Embedding Layer
        c = self.context_LSTM((c_word, c_lens))[0]
        q = self.context_LSTM((q_word, q_lens))[0]

        c = F.pad(c, (0,0,0,c_maxlen-c.size(1)))
        q = F.pad(q, (0,0,0,q_maxlen-q.size(1)))

        # 4. Attention Flow Layer
        g = att_flow_layer(c, q, c_mask,q_mask)

        # 5. Modeling Layer

       
        assert c_lens.size(0) == c_word.size(0),f'c_lens size {c_lens.size()} c_word size {c_word.size()}'

        m = self.modeling_LSTM1((g, c_lens))[0]

        m = self.modeling_LSTM2((m, c_lens))[0]
        
        m_padded = F.pad(m, (0,0,0,g.size(1) - m.size(1)))
        
        # 6. Output Layer
        p1, p2 = output_layer(g, m_padded, c_lens)
        
        p1_padded = F.pad(p1,(0,c_maxlen - p1.size()[-1]))
        p2_padded = F.pad(p2,(0,c_maxlen - p2.size()[-1]))
        # (batch, c_len), (batch, c_len)
        return p1_padded, p2_padded


# In[4]:


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += "\nquestion_text: %s" % (
            self.question_text)
        s += "\ndoc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


# In[5]:


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    entry_num = 0
    paragraph_num = 0
    qas_num = 0
    
    examples = []
    for entry in input_data:
        entry_num = entry_num + 1
        for paragraph in entry["paragraphs"]:
            paragraph_num = paragraph_num + 1
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_num = qas_num + 1
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    print(f'entry num {entry_num}')
    print(f'paragraph num {paragraph_num}')
    print(f'qas num {qas_num}')
    return examples


# In[6]:


class InputFeatures(object):
    """A single set of features of data."""
    
    
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 query_tokens,
                 doc_tokens,
                 query_length,
                 doc_length,
                 token_to_orig_map,
                 token_is_max_context,
                 query_input_ids,
                 query_input_mask,
                 doc_input_ids,
                 doc_input_mask,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.query_tokens=query_tokens
        self.doc_tokens=doc_tokens
        self.query_length=query_length
        self.doc_length=doc_length
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.query_input_ids=query_input_ids
        self.query_input_mask=query_input_mask
        self.doc_input_ids=doc_input_ids
        self.doc_input_mask=doc_input_mask
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)
def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


# In[7]:


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            
            c_len = result.start_logits.size(0)
            ls = nn.LogSoftmax(dim=0)
            mask = (torch.ones(c_len, c_len) * float('-inf')).tril(-1)
            score = (ls(result.start_logits).unsqueeze(1) + ls(result.end_logits).unsqueeze(0)) + mask
            score, start_index = score.max(dim=0)
            score, end_index = score.max(dim=0)
            start_index = torch.gather(start_index, 0, end_index).squeeze()
            start_index = start_index.tolist()
            end_index = end_index.tolist()
            
            
            if start_index >= len(feature.doc_tokens):
                continue
            if end_index >= len(feature.doc_tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue 
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
            
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.doc_tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
                
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


# In[8]:


def convert_examples_to_features(examples, tokenizer, max_query_length,max_seq_length,
                                 doc_stride, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    
    
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[0:max_query_length-1]
            
        
        query_tokens.insert(0,"[CLS]")
        query_length = len(query_tokens)

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)


        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_seq_length - 1:
                length = max_seq_length - 1
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            doc_length = doc_span.length + 1
            doc_tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            doc_segment_ids = []
            doc_tokens.append("[CLS]")

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(doc_tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(doc_tokens)] = is_max_context
                doc_tokens.append(all_doc_tokens[split_token_index])


            query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
            doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
            

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            query_input_mask = [1] * query_length
            doc_input_mask = [1] * doc_length

            # Zero-pad up to the sequence length.
            while len(query_input_ids) < max_query_length:
                query_input_ids.append(0)
                query_input_mask.append(0)

            while len(doc_input_ids) < max_seq_length:
                doc_input_ids.append(0)
                doc_input_mask.append(0)                

            assert len(query_input_ids) == max_query_length
            assert len(query_input_mask) == max_query_length
            
            assert len(doc_input_ids) == max_seq_length
            assert len(doc_input_mask) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    continue
                else:
                    doc_offset = 1
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < 5:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("query_tokens: %s" % " ".join(query_tokens))
                logger.info("doc_tokens: %s" % " ".join(doc_tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info(
                    "query_input_mask: %s" % " ".join([str(x) for x in query_input_mask]))
                logger.info("query_input_ids: %s" % " ".join([str(x) for x in query_input_ids]))
                logger.info("doc_input_ids: %s" % " ".join([str(x) for x in doc_input_ids]))
                logger.info(
                    "doc_input_mask: %s" % " ".join([str(x) for x in doc_input_mask]))
                logger.info(
                    "query_length: %d" % (query_length))
                logger.info(
                    "doc_length: %d" % (doc_length))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    query_tokens=query_tokens,
                    doc_tokens=doc_tokens,
                    query_length=query_length,
                    doc_length=doc_length,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    query_input_ids=query_input_ids,
                    query_input_mask=query_input_mask,
                    doc_input_ids=doc_input_ids,
                    doc_input_mask=doc_input_mask,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


# In[9]:


def test(bert,model,eval_examples,eval_features,args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    all_query_ids = torch.tensor([f.query_input_ids for (i,f) in enumerate(eval_features)], dtype=torch.long)
    all_query_mask = torch.tensor([f.query_input_mask for (i,f) in enumerate(eval_features)], dtype=torch.long)
    all_query_length = torch.tensor([f.query_length for (i,f) in enumerate(eval_features)], dtype=torch.long)

    all_doc_ids = torch.tensor([f.doc_input_ids for (i,f) in enumerate(eval_features)], dtype=torch.long)
    all_doc_mask = torch.tensor([f.doc_input_mask for (i,f) in enumerate(eval_features)], dtype=torch.long)
    all_doc_length = torch.tensor([f.doc_length for (i,f) in enumerate(eval_features)], dtype=torch.long)

    all_example_index = torch.arange(all_query_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_query_ids, all_query_mask,all_query_length, all_doc_ids,all_doc_mask,all_doc_length, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    bert.eval()
    model.eval()
    all_results = []
    loss = 0
    logger.info("Start evaluating")
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
        query_input_ids, query_input_mask, query_length,doc_input_ids, doc_input_mask, doc_length,example_indices= batch

        
        with torch.no_grad():
            query_embeddings, _ = bert(query_input_ids, token_type_ids=None, attention_mask=query_input_mask,output_all_encoded_layers=True)
            doc_embeddings, _ = bert(doc_input_ids, token_type_ids=None, attention_mask=doc_input_mask,output_all_encoded_layers=True)
            query_embeddings = torch.cat(query_embeddings[-4:],-1)
            doc_embeddings = torch.cat(doc_embeddings[-4:],-1)
            batch_start_logits,batch_end_logits = model(doc_embeddings.float(),query_embeddings.float(),doc_length.float(),query_length.float(),doc_input_mask.float(),query_input_mask.float())


        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu()
            end_logits = batch_end_logits[i].detach().cpu()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                      args.version_2_with_negative, args.null_score_diff_threshold)
    setattr(args, 'prediction_file', f'{args.output_dir}predictions.json')
    setattr(args, 'dataset_file', f'{args.predict_file}')
    results = evaluate.main(args)
    return results['exact_match'], results['f1']


# In[10]:


#initialize
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    
if not args.do_train and not args.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

if args.do_train:
    if not args.train_file:
        raise ValueError(
            "If `do_train` is True, then `train_file` must be specified.")
if args.do_predict:
    if not args.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified.")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

print('start loading tokenizer')
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
print('end loading tokenizer')

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
bert = BertModel.from_pretrained(args.bert_model)
bert.to(device)
model = BiDAF(args).to(device)
model.load_state_dict(torch.load(args.saved_model))
if args.local_rank != -1:
    bert = torch.nn.parallel.DistributedDataParallel(bert, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
elif n_gpu > 1:
    bert = torch.nn.DataParallel(bert)
    model = torch.nn.DataParallel(model)


# In[11]:


if args.do_train:
    train_examples = read_squad_examples(
        input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
    num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_query_length=args.max_query_length,
        doc_stride=args.doc_stride,
        is_training=True)
    
if args.do_predict:
    eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
    eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)


# In[12]:


all_query_ids = torch.tensor([f.query_input_ids for (i,f) in enumerate(train_features)], dtype=torch.long)
all_query_mask = torch.tensor([f.query_input_mask for (i,f) in enumerate(train_features)], dtype=torch.long)
all_query_length = torch.tensor([f.query_length for (i,f) in enumerate(train_features)], dtype=torch.long)

all_doc_ids = torch.tensor([f.doc_input_ids for (i,f) in enumerate(train_features)], dtype=torch.long)
all_doc_mask = torch.tensor([f.doc_input_mask for (i,f) in enumerate(train_features)], dtype=torch.long)
all_doc_length = torch.tensor([f.doc_length for (i,f) in enumerate(train_features)], dtype=torch.long)

all_start_positions = torch.tensor([f.start_position for (i,f) in enumerate(train_features)], dtype=torch.long)
all_end_positions = torch.tensor([f.end_position for (i,f) in enumerate(train_features)], dtype=torch.long)

all_example_index = torch.arange(all_query_ids.size(0), dtype=torch.long)


#all_start_positions = torch.tensor([f.start_position for (i,f) in enumerate(train_features) if i < 500], dtype=torch.long)
#all_end_positions = torch.tensor([f.end_position for (i,f) in enumerate(train_features) if i < 500], dtype=torch.long)
train_data = TensorDataset(all_query_ids, all_query_mask,all_query_length, all_doc_ids,all_doc_mask,all_doc_length, all_example_index,all_start_positions,all_end_positions)


# In[13]:


train_sampler = RandomSampler(train_data)


# In[14]:


train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
BiDAF_parameters = filter(lambda p: p.requires_grad, model.parameters())

bert_param_optimizer = list(bert.named_parameters())
bert_param_optimizer = [n for n in bert_param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': list(BiDAF_parameters), 'weight_decay': 0.01},
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
criterion = nn.CrossEntropyLoss()
#optimizer = BertAdam(parameters,
#                     lr=args.learning_rate,
#                     warmup=args.warmup_proportion,
#                     t_total=num_train_optimization_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)

# In[16]:


model.train()
bert.train()
setattr(args, 'model_time', strftime('%Y.%m.%d-%H:%M:%S',localtime()))
loss, last_epoch = 0, -1
max_dev_exact, max_dev_f1 = -1, -1
print('totally {} epoch'.format(args.epoch))    
sys.stdout.flush()
for epochs in trange(int(args.epoch), desc="Epoch"):
    print(f'epoch {epochs}')
    for i, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
        query_input_ids, query_input_mask, query_length,doc_input_ids, doc_input_mask, doc_length,example_indices,start_positions,end_positions = batch

        prev_time = time.time()  

        with torch.no_grad():
            query_embeddings, _ = bert(query_input_ids, token_type_ids=None, attention_mask=query_input_mask,output_all_encoded_layers=True)
            doc_embeddings, _ = bert(doc_input_ids, token_type_ids=None, attention_mask=doc_input_mask,output_all_encoded_layers=True)
            query_embeddings = torch.cat(query_embeddings[-4:],-1)
            doc_embeddings = torch.cat(doc_embeddings[-4:],-1)
        p1,p2 = model(doc_embeddings.float(),query_embeddings.float(),doc_length.float(),query_length.float(),doc_input_mask.float(),query_input_mask.float())
        optimizer.zero_grad()
        batch_loss = criterion(p1, start_positions) + criterion(p2, end_positions)
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
        if (i + 1) % args.print_freq == 0:
            dev_exact, dev_f1 = test(bert,model,eval_examples,eval_features,args)
            c = (i + 1) // args.print_freq

            print(f'train loss: {loss:.3f}'
                  f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                max_dev_exact = dev_exact
                best_bidaf_model = copy.deepcopy(model)
                best_bert_model = copy.deepcopy(bert)

            loss = 0
            model.train()
            bert.train()
        sys.stdout.flush()
        

args.max_f1 = max_dev_f1
print(f'max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')
torch.save(best_bidaf_model.state_dict(), f'saved_models/{args.model_time}_BiDAF_finetune{args.id}__F1{args.max_f1:5.2f}.pt')
torch.save(best_bert_model.state_dict(), f'saved_models/{args.model_time}_BERT_finetune{args.id}__F1{args.max_f1:5.2f}.pt')
print('training finished!')




