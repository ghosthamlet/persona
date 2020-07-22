
import os
import json
import time
import itertools
from filelock import FileLock
from dataclasses import dataclass
from typing import Sequence

import sys
# for import parent utils
sys.path.append('../')
import utils

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'
SEP = '<SEP>'
SPE1 = '<SPE1>'
SPE2 = '<SPE2>'
PRESET_SPECIAL_TOKENS = [PAD, SOS, EOS, UNK, 
        SEP, SPE1, SPE2]


class Vocab:
    def __init__(
        self,
        vocab,
        data_path,
        special_tokens=None
    ):
        self.stoi_map = {}
        self.itos_map = {}
        self.binary_lable = dict(
            positive=1,
            negative=0,
        )

        if special_tokens is None:
            special_tokens = PRESET_SPECIAL_TOKENS
        else:
            special_tokens = PRESET_SPECIAL_TOKENS + special_tokens

        if vocab is None:
            self.__init(data_path, special_tokens)
            return

        for k, v in vocab.items():
            self.stoi_map[k] = (v.index, v.count)
            self.itos_map[v.index] = k

        i = len(self.stoi_map)
        for k in special_tokens:
            self.stoi_map[k] = [i, 1000]
            self.itos_map[i] = k
            i += 1

    # TODO: add max_vocab_size
    def __init(
        self,
        data_path,
        special_tokens=None
    ):
        import gensim

        examples = list(ChatDataProcesser(0, 0).get_examples(data_path, 'train'))

        self.stoi_map = {}
        self.itos_map = {}

        i = 0
        for post, _, _, resp, _ in examples:
            post = list(itertools.chain(*post))
            for k in set(post + resp):
                if k not in self.stoi_map:
                    self.stoi_map[k] = [i, 0]
                    i += 1
                self.stoi_map[k][1] += 1

        min_count = 137
        self.stoi_map = {k: v 
                         for k, v in self.stoi_map.items()
                         if v[1] >= min_count and gensim.utils.RULE_DISCARD != utils.vocab_zh_trim_rule(k, v[1], min_count)}

        for i, (k, v) in enumerate(self.stoi_map.items()):
            v[0] = i

        i = len(self.stoi_map)
        for k in special_tokens:
            self.stoi_map[k] = [i, 1000]
            i += 1

        self.itos_map = {i: k for k, (i, _) in self.stoi_map.items()}

    def __len__(self):
        return len(self.stoi_map)

    def stoi(self, s):
        return self.stoi_map.get(s, self.stoi_map[UNK])[0]

    def itos(self, i):
        return self.itos_map[i]

    def binary_stoi(self, s):
        return self.binary_lable[s]

    def binary_itos(self, i):
        return [k for k, v in self.binary_lable.items() if i == v][0]


# use IterableDataset for lazy load
# shuffle and sort can't work for lazy 
# __len__ is useless for lazy load
class PersonaDataset(Dataset):
    def __init__(
        self,
        vocab,
        max_seq_length,
        data_path,
        cache_path,
        data_processer,
        mode='train',
        overwrite_cache=True,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_path,
            "cached_{}_{}".format(
                mode, str(max_seq_length),
            ),
        )
        
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                print(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                print(f"Creating features from dataset file at {data_path}")

                examples = list(data_processer.get_examples(data_path, mode))
                self.features = list(data_processer.convert_examples_to_features(
                    vocab,
                    examples,
                    mode=mode,
                ))
                start = time.time()
                # torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                print("Saving features into cached file %s [took %.3f s]" % (cached_features_file, time.time() - start))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class ChatDataProcesser:
    def __init__(
        self,
        max_seq_length,
        max_context_size,
        limit_length=None,
        complete_persona=True
    ):
        # context length <= max_seq_length * max_context_size
        self.max_seq_length = max_seq_length
        self.max_context_size = max_context_size
        self.complete_persona = complete_persona
        self.limit_length = limit_length

    def get_examples(self, path, mode):
        file_path = os.path.join(path, mode + '.txt')
        # prefer city to province, as city can infer province, inverse can't
        parse_loc = lambda x: x != '' and x.split()[-1] or UNK
        parse_gender = lambda x: '男' if x == 'male' else '女'

        with open(file_path) as f:
            i = 0
            for line in f:
                if self.limit_length is not None and i == self.limit_length:
                    break
                i += 1

                obj = json.loads(line)
                dialogs = obj['dialog']
                d_len = len(dialogs)
                persona1 = obj['profile'][0]
                persona2 = obj['profile'][1]

                if d_len == 0:
                    continue
                if self.complete_persona:
                    if not all((persona1['loc'], persona1['gender'],
                        persona2['loc'], persona2['gender'])):
                        continue
                # remove the last no resp post
                if d_len % 2 != 0:
                    dialogs = dialogs[:d_len-1]
                    d_len -= 1

                personas_no_tag = [
                        [parse_gender(persona1['gender']) or UNK, parse_loc(persona1['loc'])],
                        [parse_gender(persona2['gender']) or UNK, parse_loc(persona2['loc'])],
                        ]
                tags = [
                        (persona1['tag'][0] or UNK).split(';'),
                        (persona2['tag'][0] or UNK).split(';'),
                        ]
                persona = [('性别', parse_gender(persona2['gender']) or UNK), 
                           ('地址', parse_loc(persona2['loc'])),
                           ('兴趣', ),
                           (v for v in (persona2['tag'][0] or UNK).split(';'))]
                for i in range(0, d_len, 2):
                    if dialogs[i] == '':
                        dialogs[i] = UNK
                    context = [v[0].split() for v in dialogs[:i+1]]
                    resp = dialogs[i+1][0].split()

                    yield context, personas_no_tag, tags, resp, persona

    def convert_examples_to_features(
        self,
        vocab,
        examples,
        mode
    ):
        for context, personas_no_tag, tags, resp, persona in examples:
            icontext = [[vocab.stoi(k) for k in post[:self.max_seq_length]] 
                        + [vocab.stoi(SEP)]
                        for post in context[:self.max_context_size]]
            l = len(icontext)
            isegs = [[vocab.stoi(SPE1)] * len(icontext[i]) 
                     + [vocab.stoi(SPE2)] * (len(icontext[i+1]) if i+1 < l else 0)
                    for i in range(0, l, 2)]
            iresp = [vocab.stoi(SOS)] + [vocab.stoi(k) for k in resp[:self.max_seq_length]] + [vocab.stoi(EOS)]
            ipersonas_no_tag = list(map(lambda x: list(map(vocab.stoi, x)), personas_no_tag))
            itags = list(map(lambda x: list(map(vocab.stoi, x)), tags))
            ipersona = list(map(lambda x: list(map(vocab.stoi, x)), persona))
          # print()
          # print('context:')
          # print([vocab.itos(v) for v in list(itertools.chain(*icontext))])
          # print('segs:')
          # print([vocab.itos(v) for v in list(itertools.chain(*isegs))])
          # print('personas_no_tag:')
          # print([vocab.itos(v) for v in ipersonas_no_tag[0]])
          # print([vocab.itos(v) for v in ipersonas_no_tag[1]])
          # print('tags:')
          # print([vocab.itos(v) for v in itags[0]])
          # print([vocab.itos(v) for v in itags[1]])
          # print('resp:')
          # print([vocab.itos(v) for v in iresp])

            icontext = list(itertools.chain(*icontext))
            yield (icontext, list(itertools.chain(*isegs)), 
                    ipersonas_no_tag, itags, 
                    iresp, list(itertools.chain(*ipersona)), 
                    icontext + iresp)


class LMDataProcesser:
    def __init__(
        self,
        max_seq_length,
        limit_length=None
    ):
        self.max_seq_length = max_seq_length
        self.limit_length = limit_length

    def get_examples(self, path, mode):
        file_path = os.path.join(path, mode + '.txt')

        with open(file_path) as f:
            datas = list(f.read())
        l = len(datas)

        # self.max_seq_length-1 for every seq start with prev seq end char
        i = 0
        for j in range(1, l, self.max_seq_length-1):
            if self.limit_length is not None and i == self.limit_length:
                break
            i += 1

            e = j + self.max_seq_length
            # remove last short seq
            if e > l:
                break

            # j-1 for every seq start with prev seq end char except zero seq
            yield datas[j-1:e]

    def convert_examples_to_features(
        self,
        vocab,
        examples,
        mode
    ):
        for seq in examples:
            yield [vocab.stoi(k) for k in seq]
 
 
@dataclass
class LMFeature:
    __slots__ = ['x', 'y', 'x_mask']
    
    x: Tensor
    y: Tensor

    x_mask: Tensor
       

@dataclass
class ChatFeature:
    __slots__ = ['context', 'segs', 'personas_no_tag', 'tags',
            'resp', 'persona', 
            'context_pad_mask', 'resp_mask', 
            'resp_pad_mask', 'persona_pad_mask']

    context: Tensor
    segs: Tensor
    personas_no_tag: Tensor
    tags: Tensor

    resp: Tensor
    persona: Tensor

    context_pad_mask: Tensor
    resp_mask: Tensor
    resp_pad_mask: Tensor
    persona_pad_mask: Tensor

    lm: LMFeature


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html?highlight=collate_fn
def generate_batch(batch, pad_idx):
    context, segs, personas_no_tag, tags, resp, persona, lm = zip(*batch)
    personas_no_tag = torch.tensor(personas_no_tag).permute(1, 2, 0)

    fn = lambda x: list(map(torch.tensor, x)) 
    context_pad = pad_sequence(fn(context), padding_value=pad_idx)
    segs_pad = pad_sequence(fn(segs), padding_value=pad_idx)
    tags = itertools.chain(*tags)
    tags_pad = pad_sequence(fn(tags), padding_value=pad_idx)
    tags_pad = tags_pad.view(-1, 2, int(tags_pad.shape[1]/2)).transpose(1, 0)
    resp_pad = pad_sequence(fn(resp), padding_value=pad_idx)
    src_pad_mask = (context_pad == pad_idx).T
    tgt_pad_mask = (resp_pad == pad_idx).T
    tgt_mask = _generate_square_subsequent_mask(resp_pad.shape[0])
    persona_pad = pad_sequence(fn(persona), padding_value=pad_idx)
    persona_pad_mask = (persona_pad == pad_idx).T

    lm = generate_lm_batch(lm)

    return ChatFeature(
            context=context_pad,
            segs=segs_pad,
            personas_no_tag=personas_no_tag,
            tags=tags_pad,

            resp=resp_pad,
            persona=persona_pad,

            context_pad_mask=src_pad_mask,
            resp_mask=tgt_mask,
            resp_pad_mask=tgt_pad_mask,
            persona_pad_mask=persona_pad_mask,

            lm=lm,
    )


def generate_lm_batch(batch):
    batch = torch.tensor(batch).T
    x = batch[:-1]
    y = batch[1:]
    x_mask = _generate_square_subsequent_mask(x.shape[0])

    return LMFeature(x=x, y=y, x_mask=x_mask)


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def retokenize(fname):
    import jieba
    res = []
    with open(fname) as f:
        ss = f.read()
        for line in ss.split('\n'):
            line = line.replace(' ', '')
            line = list(jieba.cut(line))
            line = ' '.join(line)
            res.append(line)

    name, ext = os.path.splitext(fname) 
    fname_res = name + '_retoken' + ext
    with open(fname_res, 'w') as f:
        ss = '\n'.join(res)
        f.write(ss)


def build_corpus(raw_fname, corpus_fname):
    datas = []
    with open(raw_fname) as f:
        for line in f:
            datas.append(json.loads(line))

    datas_str = [] 
    for v in datas: 
        for d in v['dialog']: 
            datas_str.append(d[0]) 
        if v['profile'][0]['tag'][0] != '' or v['profile'][1]['tag'][0] != '': 
            datas_str.append((v['profile'][0]['tag'][0].replace(';', ' ') 
                + ' ' + v['profile'][1]['tag'][0].replace(';', ' ')).strip()) 

    with open(corpus_fname) as f:
        f.write('\n'.join(datas_str))


