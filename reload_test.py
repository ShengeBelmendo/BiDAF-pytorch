import json
import os
import nltk
import torch
import json
import argparse

import torch.nn as nn
import torch.nn.functional as F


from utils.nn import LSTM, Linear
from torchtext import data
from model.model import BiDAF
from torchtext import datasets
from torchtext.vocab import GloVe
from time import localtime, strftime
from model.data import SQuAD
from model.ema import EMA
import evaluate


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, args):
        path = '.data/squad'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'

        self.RAW = data.RawField()
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            dev_examples = torch.load(dev_examples_path)

            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building splits...")
            self.dev = data.TabularDataset(
                path=path + f'/dev-v1.1.jsonl',
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.dev.examples, dev_examples_path)

        print("building vocab...")
        self.CHAR.build_vocab(self.dev, min_freq=10000)
        self.WORD.build_vocab(self.dev, vectors=GloVe(name='6B', dim=args.word_dim), max_size=80000)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.dev_iter = \
            data.BucketIterator(self.dev,
                                batch_size=60,
                                device=device,
                                sort=True,
                                sort_key=lambda x: len(x.c_word))

def test(model, args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    for batch in iter(data.dev_iter):
        with torch.no_grad():
            p1, p2 = model(batch.c_char,batch.q_char,batch.c_word[0],batch.q_word[0],batch.c_word[1],batch.q_word[1])
        #p1, p2 = model(batch)
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()

        # (batch, c_len, c_len)
        batch_size, c_len = p1.size()
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

        for i in range(batch_size):
            id = batch.id[i]
            answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
            answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
            answers[id] = answer

    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluate.main(args)
    return loss, results['exact_match'], results['f1']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-channel-width', default=8, type=int)
    parser.add_argument('--char-channel-size', default=200, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()
 
    print(args.model_path)
    
    #bdata = SQuAD(args)

    setattr(args, 'char_vocab_size', len(bdata.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(bdata.WORD.vocab))
    setattr(args, 'max_f1', 0)
    setattr(args, 'dataset_file', f'.data/squad/{args.dev_file}')
    setattr(args, 'model_time', strftime('%Y.%m.%d-%H:%M:%S', localtime()))
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    args.prediction_file = 'predictions/' + args.model_time + '_' + 'prediction.out'
    print('data loading complete!')

    #device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    #model = BiDAF(args, bdata.WORD.vocab.vectors)
    print(f'{args.model_path}')
    print(str(args.model_path))
    #model.load_state_dict(torch.load(str(args.model_path)))
    #model.to(device)
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

    #model.load_state_dict(torch.load(args.model_path))
    #model.eval()
    #dev_loss, dev_exact, dev_f1 = test(model, args, data)

    #print(f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

if __name__ == '__main__':
    main()
