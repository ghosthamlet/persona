
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


class ChatDataProcesser:
    def __init__(
        self,
        max_seq_length,
        max_context_size,
        limit_length=None,
        complete_persona=True
    ):
        assert max_seq_length is not None
        assert max_context_size is not None

        # context length <= max_seq_length * max_context_size
        self.max_seq_length = max_seq_length
        self.max_context_size = max_context_size
        self.complete_persona = complete_persona
        self.limit_length = limit_length

    def get_examples(self, path, mode):
        file_path = os.path.join(path, mode + '.txt')
        char_emb = True
        if char_emb:
            parse_loc = lambda x: x != '' and x or UNK
        else:
            # prefer city to province, as city can infer province, inverse can't
            parse_loc = lambda x: x != '' and x.split()[-1] or UNK
        parse_gender = lambda x: '男' if x == 'male' else '女'

        with open(file_path) as f:
            cnt = 0
            for line in f:
                if self.limit_length is not None and cnt == self.limit_length:
                    break
                cnt += 1

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
                dialogs = dialogs[:self.max_context_size]
                d_len = len(dialogs)

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
        char_emb = True
        for context, personas_no_tag, tags, resp, persona in examples:
            icontext = [[vocab.stoi(k) for k in post[:self.max_seq_length]] 
                        + [vocab.stoi(SEP)]
                        for post in context]
            l = len(icontext)
            isegs = [[vocab.stoi(SPE1)] * len(icontext[i]) 
                     + [vocab.stoi(SPE2)] * (len(icontext[i+1]) if i+1 < l else 0)
                    for i in range(0, l, 2)]
            iresp = [vocab.stoi(SOS)] + [vocab.stoi(k) 
                    for k in resp[:self.max_seq_length]] + [vocab.stoi(EOS)]
            if char_emb:
                ipersonas_no_tag = list(map(lambda x: list(map(vocab.stoi, ''.join(x))), personas_no_tag))
                itags = list(map(lambda x: list(map(vocab.stoi, ''.join(x))), tags))
                ipersona = list(map(lambda x: list(map(vocab.stoi, ''.join(x))), persona))
            else:
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
        seq_length = self.max_seq_length
        limit_length = self.limit_length
        char_emb = True

        with open(file_path) as f:
            datas = f.read()
        if char_emb:
            if limit_length is not None:
                datas = datas[:limit_length*seq_length]
            datas = list(datas)
        else:
            datas = datas.split()
            if limit_length is not None:
                datas = datas[:limit_length*seq_length]
        l = len(datas)

        # self.max_seq_length-1 for every seq start with prev seq end char
        cnt = 0
        for j in range(1, l, seq_length-1):
            if limit_length is not None and cnt == limit_length:
                break
            cnt += 1

            e = j + seq_length
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
    __slots__ = ['x', 'y', 'x_mask', 'x_pad_mask']
    
    x: Tensor
    y: Tensor

    x_mask: Tensor
    x_pad_mask: Tensor
       

@dataclass
class ChatFeature:
    __slots__ = ['context', 'segs', 'personas_no_tag', 
            'tags', 'resp', 'persona', 
            'context_pad_mask', 'resp_mask', 
            'resp_pad_mask', 'persona_pad_mask', 'lm']

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
    char_emb = True

    fn = lambda x: list(map(torch.tensor, x)) 
    context_pad = pad_sequence(fn(context), padding_value=pad_idx)
    segs_pad = pad_sequence(fn(segs), padding_value=pad_idx)
    tags = itertools.chain(*tags)
    tags_pad = pad_sequence(fn(tags), padding_value=pad_idx)
    tags_pad = tags_pad.view(-1, 2, int(tags_pad.shape[1]/2)).transpose(1, 0)
    resp_pad = pad_sequence(fn(resp), padding_value=pad_idx)
    src_pad_mask = (context_pad == pad_idx).T
    tgt_pad_mask = (resp_pad == pad_idx).T
    tgt_mask = utils.generate_square_subsequent_mask(resp_pad.shape[0])
    persona_pad = pad_sequence(fn(persona), padding_value=pad_idx)
    persona_pad_mask = (persona_pad == pad_idx).T
    if char_emb:
        # batch_size X n_persona X 2 
        tmp = list(map(lambda x: pad_sequence(fn(x), padding_value=pad_idx),
                personas_no_tag))
        # n_persona X batch_size X 2 --> 2 X n_persona X batch_size
        personas_no_tag_pad = pad_sequence(tmp, padding_value=pad_idx).permute(2, 0, 1) 
    else:
        personas_no_tag_pad = torch.tensor(personas_no_tag).permute(1, 2, 0)

    lm = generate_lm_batch(lm, pad_idx, in_chat=True)

    return ChatFeature(
            context=context_pad,
            segs=segs_pad,
            personas_no_tag=personas_no_tag_pad,
            tags=tags_pad,

            resp=resp_pad,
            persona=persona_pad,

            context_pad_mask=src_pad_mask,
            resp_mask=tgt_mask,
            resp_pad_mask=tgt_pad_mask,
            persona_pad_mask=persona_pad_mask,

            lm=lm,
    )


def generate_lm_batch(batch, pad_idx, in_chat=False):
    if not in_chat:
        batch = torch.tensor(batch).T
    else:
        batch = pad_sequence(list(map(torch.tensor, batch)) , padding_value=pad_idx)
    x = batch[:-1]
    y = batch[1:]
    x_mask = utils.generate_square_subsequent_mask(x.shape[0])
    x_pad_mask = (x == pad_idx).T

    return LMFeature(x=x, y=y, x_mask=x_mask, x_pad_mask=x_pad_mask)


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


