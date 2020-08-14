
import os
import json
import time
import random
import itertools
from dataclasses import dataclass
from typing import Sequence

import sys
# for import parent utils
sys.path.append('../')
import utils
from utils import UNK, SEP, SOS, EOS, SPE1, SPE2, CLS

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

_USE_BERT_FEATURE = False


class ChatVocab:
    def __init__(
        self,
        tokenizer
    ):
        self.tokenizer = tokenizer

    def __len__(self):
        # XXX: tokenizer.vocab_size still not include custom tokens, must use len(tokenizer)
        return len(self.tokenizer)

    def stoi(self, s):
        i = self.tokenizer.convert_tokens_to_ids(s)
        # default unk token is changed to UNK, 
        # so tokenizer can't get it, returned None
        if i is None:
            i = self.tokenizer.convert_tokens_to_ids(UNK)
        return i

    def itos(self, i):
        return self.tokenizer.convert_ids_to_tokens(i)
 
 
class PersonaVocab:
    def __init__(
        self,
        vocab_fname,
        special_tokens=None
    ):
        self.stoi_map = {}
        self.itos_map = {}
        if special_tokens is None:
            special_tokens = utils.PRESET_SPECIAL_TOKENS
        else:
            special_tokens = utils.PRESET_SPECIAL_TOKENS + special_tokens
        
        with open(vocab_fname) as f:
            i = 0
            for line in f:
                k, cnt = line.split('\t')
                self.stoi_map[k] = (i, cnt)
                self.itos_map[i] = k
                i += 1

        i = len(self.stoi_map)
        for k in special_tokens:
            self.stoi_map[k] = [i, 1000]
            self.itos_map[i] = k
            i += 1

    def __len__(self):
        return len(self.stoi_map)

    def stoi(self, s):
        return self.stoi_map.get(s, self.stoi_map[UNK])[0]

    def itos(self, i):
        return self.itos_map[i]
 
 
class ChatDataProcesser:
    def __init__(
        self,
        max_seq_length,
        max_context_size,
        vocab,
        persona_vocab=None,
        limit_length=None,
        complete_persona=True,
        tokenizer=None,
    ):
        assert max_seq_length is not None
        assert max_context_size is not None

        # context length <= max_seq_length * max_context_size
        self.max_seq_length = max_seq_length
        self.max_context_size = max_context_size
        self.complete_persona = complete_persona
        self.limit_length = limit_length
        self.vocab = vocab
        self.persona_vocab = vocab
        self.tokenizer = tokenizer

    def get_examples(self, path, mode):
        file_path = os.path.join(path, mode + '.txt')
        char_emb = True
        _tok = self.tokenizer
        tokenizer = lambda x: _tok(x.replace(' ', '')) if _tok is not None else x.split()
        if char_emb:
            parse_loc = lambda x: tokenizer(x != '' and ' '.join(x) or UNK)
        else:
            # prefer city to province, as city can infer province, inverse can't
            parse_loc = lambda x: [x != '' and x.split()[-1] or UNK]
        if self.persona_vocab is not None:
            parse_loc = lambda x: [x != '' and x or UNK]
        parse_gender = lambda x: ['男' if x == 'male' else ('女' if x == 'female' else UNK)]

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
                d_len = len(dialogs)

                personas_no_tag = [
                        parse_gender(persona1['gender']) + parse_loc(persona1['loc']),
                        parse_gender(persona2['gender']) + parse_loc(persona2['loc']),
                        ]
                if self.persona_vocab is not None:
                    tags = [
                            (persona1['tag'][0] or UNK).split(';'),
                            (persona2['tag'][0] or UNK).split(';'),
                            ]
                    persona = [['性别'] + parse_gender(persona2['gender']), 
                               ['地址'] + parse_loc(persona2['loc']),
                               ['兴趣'] + (persona2['tag'][0] or UNK).split(';')]              
                else:
                    tags = [
                            list(itertools.chain(*[tokenizer(' '.join(v)) for v in (persona1['tag'][0] or UNK).split(';')])),
                            list(itertools.chain(*[tokenizer(' '.join(v)) for v in (persona2['tag'][0] or UNK).split(';')])),
                            ]
                    persona = [tokenizer('性 别') + parse_gender(persona2['gender']), 
                               tokenizer('地 址') + parse_loc(persona2['loc']),
                              list(itertools.chain(*(tokenizer('兴 趣') 
                                  + [tokenizer(' '.join(v)) for v in (persona2['tag'][0] or UNK).split(';')])))]
                for i in range(0, d_len, 2):
                    if dialogs[i] == '':
                        dialogs[i] = UNK
                    context = [tokenizer(v[0]) for v in dialogs[:i+1][-(self.max_context_size+1):]]
                    resp = tokenizer(dialogs[i+1][0])

                    # print(context, personas_no_tag, tags, resp, persona)
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
                        for post in context]
            if _USE_BERT_FEATURE:
                icontext[0] = [vocab.stoi(CLS)] + icontext[0]
            l = len(icontext)
            isegs = [[vocab.stoi(SPE1)] * len(icontext[i]) 
                     + [vocab.stoi(SPE2)] * (len(icontext[i+1]) if i+1 < l else 0)
                    for i in range(0, l, 2)]
            if _USE_BERT_FEATURE:
                iresp = [vocab.stoi(CLS)] + [vocab.stoi(k) 
                        for k in resp[:self.max_seq_length]] + [vocab.stoi(SEP)]
            else:
                iresp = [vocab.stoi(SOS)] + [vocab.stoi(k) 
                        for k in resp[:self.max_seq_length]] + [vocab.stoi(EOS)]
            _vocab = self.vocab
            if self.persona_vocab is not None:
                _vocab = self.persona_vocab
            ipersonas_no_tag = list(map(lambda x: list(map(_vocab.stoi, x)), personas_no_tag))
            itags = list(map(lambda x: list(map(_vocab.stoi, x)), tags))
            ipersona = list(map(lambda x: list(map(_vocab.stoi, x)), persona))
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
                    # XXX: for lm auxiliary task
                    # better performance
                    # icontext + iresp)
                    # fast and use fewer memory
                    iresp)


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
    __slots__ = ['x', 'y', 'x_mlm', 
            'x_mask', 'x_pad_mask',
            'x_mlm_pad_mask']
    
    x: Tensor
    y: Tensor
    x_mlm: Tensor
    # y_mlm: Tensor

    x_mask: Tensor
    x_pad_mask: Tensor
    x_mlm_pad_mask: Tensor


@dataclass
class ChatFeature:
    __slots__ = ['context', 'segs', 'personas_no_tag', 
            'tags', 'resp', 'persona', 
            'context_pad_mask', 'resp_mask', 
            'resp_pad_mask', 'persona_pad_mask',
            'personas_no_tag_pad_mask', 'tags_pad_mask', 
            'post', 'post_pad_mask', 'lm']

    # context include post
    context: Tensor
    # for some calc
    post: Tensor
    segs: Tensor
    personas_no_tag: Tensor
    tags: Tensor

    resp: Tensor
    persona: Tensor

    context_pad_mask: Tensor
    post_pad_mask: Tensor
    resp_mask: Tensor
    resp_pad_mask: Tensor
    persona_pad_mask: Tensor

    # for pretrain_feature
    personas_no_tag_pad_mask: Tensor
    tags_pad_mask: Tensor

    lm: LMFeature


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html?highlight=collate_fn
def generate_batch(batch, vocab, persona_vocab):
    pad_idx = vocab.stoi(utils.PAD)
    persona_pad_idx = pad_idx
    if persona_vocab is not None:
        persona_pad_idx = persona_vocab.stoi(utils.PAD)
    context, segs, personas_no_tag, tags, resp, persona, lm = zip(*batch)
    char_emb = True

    post = []
    cls_idx = vocab.stoi(utils.CLS)
    sep_idx = vocab.stoi(utils.SEP)
    for v in context:
        post_start = list(reversed(v)).index(sep_idx)
        if _USE_BERT_FEATURE:
            # cls_idx for bert-like sentence rep, xlnet is last token
            post.append([cls_idx] + v[-post_start:])
        else:
            post.append(v[-post_start:-1])

    fn = lambda x: list(map(torch.tensor, x)) 
    context_pad = pad_sequence(fn(context), padding_value=pad_idx)
    post_pad = pad_sequence(fn(post), padding_value=pad_idx)
    segs_pad = pad_sequence(fn(segs), padding_value=pad_idx)
    tags = itertools.chain(*tags)
    tags_pad = pad_sequence(fn(tags), padding_value=persona_pad_idx)
    tags_pad = tags_pad.view(-1, 2, int(tags_pad.shape[1]/2)).transpose(1, 0)
    resp_pad = pad_sequence(fn(resp), padding_value=pad_idx)
    persona_pad = pad_sequence(fn(persona), padding_value=persona_pad_idx)

    context_pad_mask = (context_pad == pad_idx).T
    post_pad_mask = (post_pad == pad_idx).T
    tags_pad_mask = (tags_pad == persona_pad_idx).T
    resp_mask = utils.generate_square_subsequent_mask(resp_pad.shape[0])
    resp_pad_mask = (resp_pad == pad_idx).T
    persona_pad_mask = (persona_pad == persona_pad_idx).T
    if char_emb:
        # batch_size X n_persona X 2 
        tmp = list(map(lambda x: pad_sequence(fn(x), padding_value=persona_pad_idx),
                personas_no_tag))
        # n_persona X batch_size X 2 --> 2 X n_persona X batch_size
        personas_no_tag_pad = pad_sequence(tmp, padding_value=persona_pad_idx).permute(2, 0, 1) 
    else:
        personas_no_tag_pad = torch.tensor(personas_no_tag).permute(1, 2, 0)
    personas_no_tag_pad_mask = (personas_no_tag_pad == persona_pad_idx).T

    lm = generate_lm_batch(lm, vocab, in_chat=True)

    return ChatFeature(
            context=context_pad,
            post=post_pad,
            segs=segs_pad,
            personas_no_tag=personas_no_tag_pad,
            tags=tags_pad,

            resp=resp_pad,
            persona=persona_pad,

            context_pad_mask=context_pad_mask,
            post_pad_mask=post_pad_mask,
            resp_mask=resp_mask,
            resp_pad_mask=resp_pad_mask,
            persona_pad_mask=persona_pad_mask,

            personas_no_tag_pad_mask=personas_no_tag_pad_mask,
            tags_pad_mask=tags_pad_mask,

            lm=lm,
    )


def generate_lm_batch(batch, vocab, in_chat=False):
    pad_idx = vocab.stoi(utils.PAD)
    mask_idx = vocab.stoi(utils.MASK)

    if not in_chat:
        x_mlm = None
        x_mlm_pad_mask = None
        batch = torch.tensor(batch).T
    else:
        x_mlm = []
        for v in batch:
            l = len(v)
            if l == 3:
                # bos + w + eos
                x_mlm.append(v)
                continue
            mask = 1
            if l > 5:
                # mask = 2
                # 99/100 is 2, 1/100 is 0
                mask = min(2, random.randint(0, 99))
                if mask == 1:
                    mask = 2
            start = random.randint(1, l-mask-1)
            x_mlm.append(v[0:start] + [mask_idx] + v[start+mask:])

        batch = pad_sequence(list(map(torch.tensor, batch)) , padding_value=pad_idx)
        x_mlm = pad_sequence(list(map(torch.tensor, x_mlm)) , padding_value=pad_idx)
        x_mlm_pad_mask = (x_mlm == pad_idx).T

    x = batch[:-1]
    y = batch[1:]
    x_mask = utils.generate_square_subsequent_mask(x.shape[0])
    x_pad_mask = (x == pad_idx).T

    return LMFeature(x=x, y=y, x_mlm=x_mlm, 
            x_mask=x_mask, x_pad_mask=x_pad_mask,
            x_mlm_pad_mask=x_mlm_pad_mask)


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

