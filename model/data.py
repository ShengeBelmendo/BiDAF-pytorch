import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) in [0x202F, 0x3000, 0x2009]:
        return True
    return False


def word_tokenize(tokens):
    return [character for character in tokens if not is_whitespace(character)]


class SQuAD():
    def __init__(self, args):
        path = '.data/cmrc'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'

        print("preprocessing data files...")
        if not os.path.exists(f'{path}/{args.train_file}l'):
            self.preprocess_file(f'{path}/{args.train_file}')
        if not os.path.exists(f'{path}/{args.dev_file}l'):
            self.preprocess_file(f'{path}/{args.dev_file}')

        self.RAW = data.RawField()
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': ('c_word', self.WORD),
                       'question': ('q_word', self.WORD)}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD),
                       ('q_word', self.WORD)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.dev = data.TabularDataset.splits(
                path=path,
                train=f'{args.train_file}l',
                validation=f'{args.dev_file}l',
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        # cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        vectors = Vectors(name="sgns.context.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5",
                          cache=".vector_cache")
        self.WORD.build_vocab(self.train, self.dev, vectors=vectors)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print("building iterators...")
        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_sizes=[args.train_batch_size, args.dev_batch_size],
                                       sort=True,
                                       device=device,
                                       sort_key=lambda x: len(x.c_word))

    def preprocess_file(self, path):
        dump = []

        context_length = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = list(context)
                    context_length.append(len(tokens))
                    clear_tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            answer_text = ''
                            answer = answer_text.join([i for i in answer if not is_whitespace(i)])
                            s_idx = ans['answer_start']
                            origin_s_idx = ans['answer_start']
                            if s_idx < 0:
                                break
                            e_idx = s_idx + len(answer) - 1

                            num = []
                            s_found = False
                            for i, t in enumerate(tokens):
                                if i < origin_s_idx:
                                    if is_whitespace(t):
                                        s_idx = s_idx - 1
                                        e_idx = e_idx - 1
                                else:
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))

        with open(f'{path}l', 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)