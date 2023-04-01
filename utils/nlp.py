import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import re
import string
import itertools
from collections import Counter
from tqdm import tqdm
import numpy as np
import io
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from datasets import load_dataset
from torch.utils.data import DataLoader

def get_data(args):

    if args['dataset'] != 'imdb':

        train_data = load_dataset(args['dataset'], split='train')
        test_data = load_dataset(args['dataset'], split='test')

        if args['dataset'] == 'dbpedia_14':
            tf = 'content'
            class_num = 14
        elif args['dataset'] == 'ag_news':
            tf = 'text'
            class_num = 4
        elif args['dataset'] == 'banking77':
            tf = 'text'
            class_num = 77
        elif args['dataset'] == 'emotion':
            tf = 'text'
            class_num = 6
        elif args['dataset'] == 'rotten_tomatoes':
            tf = 'text'
            class_num = 2
        elif args['dataset'] == 'yelp_review_full':
            tf = 'text'
            class_num = 5
        elif args['dataset'] == 'sst2':
            tf = 'sentence'
            class_num = 2
            test_data = load_dataset(args['dataset'], split='validation')

        train_text = [b[tf] for b in train_data]
        test_text = [b[tf] for b in test_data]
        train_label = [b['label'] for b in train_data]
        test_label = [b['label'] for b in test_data]
        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
        clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)

        vocab = create_vocab(clean_train)

    else:
        from sklearn.model_selection import train_test_split
        class_num = 2
        df = pd.read_csv('./IMDB_Dataset.csv')
        df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)
        corpus = [word for text in df['cleaned_reviews']
                  for word in text.split()]
        text = [t for t in df['cleaned_reviews']]
        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        vocab = create_vocab(corpus)
        clean_train, clean_test, train_label, test_label = train_test_split(
            text, label, test_size=0.2)
        clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
        clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)

    trainset = Textset(clean_train, train_label, vocab, args['max_len'])
    testset = Textset(clean_test, test_label, vocab, args['max_len'])
    train_loader = DataLoader(
        trainset, batch_size=args['train_bsz'], collate_fn=trainset.collate, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        testset, batch_size=args['test_bsz'], collate_fn=testset.collate, pin_memory=True)

    return train_loader, test_loader, class_num, vocab

def get_word_vector(vocab, emb='glove'):

    if emb == 'glove':
        fname = 'glove.6B.300d.txt'

        with open(fname, 'rt', encoding='utf8') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in tqdm(range(len(full_content)), total=len(full_content), desc='loading glove vocabs...'):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val)
                            for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings

    elif emb == 'fasttext':
        fname = 'wiki-news-300d-1M.vec'

        fin = io.open(fname, 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}

        for line in tqdm(fin, total=1000000, desc='loading fasttext vocabs...'):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in vocab.keys():
                continue
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)

    else:
        raise Exception('emb not implemented')

    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(300))

    print('found', find, 'words in', emb)
    return torch.stack(w, dim=0)

def data_cleansing(_text, _labels, doRemove=False):
    """
    Detect or remove the empty samples.
    """
    assert len(_text)==len(_labels), "Text and label list need to be the same length."

    clear_text = []
    clear_label = []
    flag = False

    for idx ,t in enumerate(_text):
        if len(t) == 0:
            flag = True
        else:
            if doRemove:
                clear_text.append(t)
                clear_label.append(_labels[idx])

    if (flag == True) and (doRemove == True):
        print("Info: Detect the empty samples, and remove them!")
        print("Size change: {0}->{1}".format(len(_text), len(clear_text)))
    elif (flag == True) and (doRemove == False):
        print("Warning: same samples in data preprocessing outputs empty list. This will damage the model.")

    if doRemove:
        return clear_text, clear_label
    else:
        return _text, _labels


def data_preprocessing(text, remove_stopword=False):

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)

    return text

def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 2

    vocab_to_int = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    print('vocab size', len(vocab_to_int))

    return vocab_to_int


class Textset(Dataset):
    def __init__(self, text, label, vocab, max_len, pad_value=0, pad_token='<pad>'):
        super().__init__()
        self.pad_value = pad_value
        self.pad_token = pad_token

        method = 1
        self.handle(text, label, vocab, max_len, method)

    def handle(self, text, label, vocab, max_len, method=1):

        if method == 0:
            print("[Textset] Using method 0")
            new_text = []
            for t in text:
                t_split = t.split(' ')
                if len(t_split) > max_len:
                    t_split = t_split[:max_len]
                    new_text.append(' '.join(t_split))
                else:
                    while len(t_split) < max_len:
                        t_split.append(self.pad_token)
                    new_text.append(' '.join(t_split))
            self.x = new_text
            self.y = label
            self.vocab = vocab
        
        elif method == 1:
            print("[Textset] Using method 1")
            new_text = []
            for t in text:
                t_split = t.split(' ')
                if len(t_split) > max_len:
                    t_split = t_split[:max_len]
                    new_text.append(' '.join(t_split))
                else:
                    new_text.append(' '.join(t_split))
            self.x = new_text
            self.y = label
            self.vocab = vocab

        elif method == 2:
            print("[Textset] Using method 2")
            new_text = []
            for t in text:
                if len(t) > max_len:
                    t = t[:max_len]
                    new_text.append(t)
                else:
                    new_text.append(t)
            self.x = new_text
            self.y = label
            self.vocab = vocab
        else:
            raise RuntimeError("Textset method setting error!")

    def collate(self, batch):

        x = [torch.tensor(x) for x, y in batch]
        y = [y for x, y in batch]
        x_tensor = pad_sequence(x, True)
        y = torch.tensor(y)
        return x_tensor, y

    def convert2id(self, text):
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r

    def __getitem__(self, idx):
        text = self.x[idx]
        word_id = self.convert2id(text)
        return word_id, self.y[idx]

    def __len__(self):
        return len(self.x)
