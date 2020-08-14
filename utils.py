
import os
import math
import random
import time
import logging
import itertools
from filelock import FileLock

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from prefetch_generator import BackgroundGenerator


def get_device(require_device):
    return torch.device('cuda' if torch.cuda.is_available() and require_device == 'cuda' else 'cpu')

 
def set_random_seed(seed, device):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :]
        return x
       
                                
class Attention(nn.Module):
    """different attention implementions
     https://nbviewer.jupyter.org/github/susanli2016/NLP-with-Python/blob/master/Attention%20Basics.ipynb
     https://github.com/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Torch.py
     https://github.com/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi-LSTM(Attention)-Torch.py
     https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py#L318
     https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html#defining-our-nn-module-and-optimizer
     https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html#define-decoders-attention-module
     https://github.com/facebookresearch/ParlAI/blob/master/projects/controllable_dialogue/controllable_seq2seq/modules.py#L815
     https://github.com/philipperemy/keras-attention-mechanism
    """
    def __init__(
        self,
        enc_hid_dim,
        dec_hid_dim,
        attn_dim
    ):
        super().__init__()

        attn_in = enc_hid_dim + dec_hid_dim
        self.attn = nn.Linear(attn_in, attn_dim)

    def forward(self, query, values):
        """
        Args:
            query:
            values:

        Shape:
            query: batch_size X dec_hid_dim
            values: seq_len X batch_size X enc_hid_dim
        """
        seq_len = values.shape[0]
        repeat_query = query.unsqueeze(1).repeat(1, seq_len, 1)
        values = values.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeat_query,
            values),
            dim=2)))
        energy = torch.sum(energy, dim=2)

        return F.softmax(energy, dim=1)


def num_dir(enc_bidi):
    return 2 if enc_bidi else 1


def embedding(
    input_dim,
    emb_dim,
    embeddings,
    emb_freeze,
    pad_idx
):
    if embeddings is None:
        return nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
    return nn.Embedding.from_pretrained(embeddings, freeze=emb_freeze, padding_idx=pad_idx)
 

def mask_seq_batch(seq, mask):
    return seq[:, mask]


def label_resp_profile_v(embs, vocab, resp_fname, profile_key_fname, profiles=None):
    """
    run first:
    embs, vocab = load_embeddings_and_vocab(vec_fname, vocab_fname)
    """
    import datasets
    import torch.nn.functional as F
  # params = dict(
  #     input_dim = embs.shape[0],
  #     emb_dim = embs.shape[1],
  #     dropout = 0,
  #     pad_idx = None,
  #     emb_freeze = True,
  #     embeddings = embs,
  # )
  # pos_detector = modules.PositionDetector(*params)
  # profile_emb = modules.ProfileEmb(*params)
  # v_pos = pos_detector(y, profile_emb(profiles[profile_key]))
    resps = []
    resps_emb = []
    with open(resp_fname) as f:
        for row in f:
            row = row.strip()
            if row != '':
                xs = row.split(' ')
                resps.append(xs)
                resps_emb.append([embs[vocab[k].index].tolist() 
                    for k in xs if k in vocab])
            else:
                print('resp empty')

    profiles_v = []
    profiles_v_emb = []
    poss = []
    if profiles is None:
        profiles = dict(
                #姓名='张',
                姓名='刘德华',
                年龄='三岁',
                性别='男孩',
                爱好='动漫',
                特长='钢琴',
                体重='60',
                地址='北京',
                星座='双子座',
        )
    with open(profile_key_fname) as f:
        for row in f:
            row = row.strip()
            if row != '':
                xs = row.split(' ')
                profiles_v.append(row)
                k = profiles[datasets.EN_TO_ZH[xs[1]]]
                profiles_v_emb.append(embs[vocab[k].index].tolist())
                poss.append(xs[0] == 'positive')
            else:
                print('key empty')

    ret = []
    for resp, profile_v, resp_emb, profile_v_emb, pos in \
            zip(resps, profiles_v, resps_emb, profiles_v_emb, poss):
        if not pos:
            continue
        resp_emb = torch.tensor(resp_emb)
        profile_v_emb = torch.tensor(profile_v_emb)
        print()
        print(resp)
        if resp_emb.shape[0] == 0:
            print('!!!!!!!!!!!!!!!! no emb !!!!!!!!!!!!!!!!!')
            continue
        sim = F.cosine_similarity(resp_emb, profile_v_emb.unsqueeze(0), dim=1)
        v_pos = sim.argmax(dim=0)
        prop = sim[v_pos]
        print(profile_v, resp[v_pos], prop)
        if prop > 0.6:
            ret.append((resp, profile_v, resp[v_pos], prop))

    return ret


# https://discuss.pytorch.org/t/print-autograd-graph/692/33
# https://github.com/waleedka/hiddenlayer/blob/master/demos/pytorch_graph.ipynb
def print_backward_graph(tensor):
    def fn(grad_fn):
        print('------------------------')
        next_functions = grad_fn.next_functions
        for v in next_functions:
            if v[0] is None:
                continue
            print(v[0], v[1])
        # print(next_functions)
        for v in next_functions:
            if v[0] is None:
                continue
            fn(v[0])
    print(tensor.grad_fn)
    fn(tensor.grad_fn)

 
def vocab_zh_trim_rule(word, count, min_count):
    import gensim
    l = len(word)
    o = ord(word[0])

    # remove single english letter, keep other single ascii
    if l == 1 and (91 > o > 64 or 123 > o > 96):
        return gensim.utils.RULE_DISCARD

    # remove english or number with 2-20 letters 
    # some special char may be removed
    if l > 1 and sum(map(ord, word)) < 123 * 20:
        return gensim.utils.RULE_DISCARD

    return gensim.utils.RULE_DEFAULT
                                   

def feature_to_device(feature, device):
    if not hasattr(feature, '__slots__'):
        return

    for k in feature.__slots__:
        v = getattr(feature, k)
        if type(v) == torch.Tensor:
            setattr(feature, k, v.to(device))
        else:
            feature_to_device(v, device)
 
         
# PAD = '<PAD>'
# SOS = '<SOS>'
# EOS = '<EOS>'
# UNK = '<UNK>'
# SEP = '<SEP>'
# SPE1 = '<SPE1>'
# SPE2 = '<SPE2>'
PAD = '[PAD]'
SOS = '[BOS]'
EOS = '[EOS]'
UNK = '[UNK]'
SEP = '[SEP]'
SPE1 = '[SPE1]'
SPE2 = '[SPE2]'
MASK = '[MASK]'
CLS = '[CLS]'
PRESET_SPECIAL_TOKENS = [PAD, SOS, EOS, UNK, 
        SEP, SPE1, SPE2, MASK, CLS]
 
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

    # TODO: read data from the exists vocab file
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
        # assert s in self.stoi_map, 'Char %s not exists!' % s

        return self.stoi_map.get(s, self.stoi_map[UNK])[0]

    def itos(self, i):
        return self.itos_map[i]

    def binary_stoi(self, s):
        return self.binary_lable[s]

    def binary_itos(self, i):
        return [k for k, v in self.binary_lable.items() if i == v][0]
       

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    ATTR_TO_SPECIAL_TOKEN = {'bos_token': SOS, 'eos_token': EOS, 'pad_token': PAD,
                             'sep_token': SEP, 'unk_token': UNK, 'cls_token': CLS,
                             'mask_token': MASK,
                             'additional_special_tokens': [SPE1, SPE2]}
    # orig_num_tokens = len(tokenizer.vocab)
    orig_num_tokens = tokenizer.vocab_size
    # XXX: tokenizer.vocab_size still not include custom tokens, must use len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
                                                     

# TODO: move vocab arg to ChatDataProcesser
# use IterableDataset for lazy load
# shuffle and sort can't work for lazy 
# __len__ is useless for lazy load
class PersonaDataset(Dataset):
    def __init__(
        self,
        vocab,
        max_seq_length,
        limit_example_length,
        data_path,
        cache_path,
        data_processer,
        mode='train',
        overwrite_cache=False,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_path,
            "cached_{}_{}_{}".format(
                mode, str(max_seq_length),
                str(limit_example_length or 'all'),
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
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                print("Saving features into cached file %s [took %.3f s]" % (cached_features_file, time.time() - start))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
       
 
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) 

 
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
     
 
def uniform_init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            n = param.data.shape[-1]
            nn.init.uniform_(param.data, -math.sqrt(3/n), math.sqrt(3/n))
            # nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def xavier_init_weights(m):
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_word2vec(corpus_fname, vec_fname, vocab_fname, 
        max_vocab_size, trim_rule=vocab_zh_trim_rule, emb_dim=100):
    """
    no need utils.vocab_zh_trim_rule for char embedding
    """
    import gensim
    lss = gensim.models.word2vec.LineSentence(corpus_fname) 
    # skip-gram is more accuracy for most words, but CBOW is better for name similarity
    model = gensim.models.Word2Vec(lss, 
            max_final_vocab=max_vocab_size, size=emb_dim,
            trim_rule=trim_rule)
    model.wv.save_word2vec_format(vec_fname, vocab_fname)
    return model


def load_embeddings_and_vocab(vec_fname, vocab_fname):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_fname, vocab_fname)
    return torch.tensor(model.vectors), model.vocab


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(feature, vocab, model, args, current_output=None):
    """Copy from https://github.com/huggingface/transfer-learning-conv-ai/blob/master/interact.py
    For beam search see: https://github.com/atselousov/transformer_chatbot/blob/agent/model/transformer_model.py

    Examples:
       >>>  history = []
       >>>  while True:
       >>>      raw_text = input(">>> ")
       >>>      while not raw_text:
       >>>          print('Prompt should not be empty!')
       >>>          raw_text = input(">>> ")
       >>>      history.append(tokenizer.encode(raw_text))
       >>>      with torch.no_grad():
       >>>          out_ids = sample_sequence(personality, history, tokenizer, model, args)
       >>>      history.append(out_ids)
       >>>      history = history[-(2*args.max_history+1):]
       >>>      out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
       >>>      print(out_text)
    """
    sos_idx = vocab.stoi(SOS)
    eos_idx = vocab.stoi(EOS)
    special_tokens_ids = [vocab.stoi(k) for k in PRESET_SPECIAL_TOKENS]
    if current_output is None:
        current_output = [[sos_idx] for _ in range(feature.context.shape[1])]

    for seq_i in range(args.max_seq_length):
        feature = build_input_from_segments(feature, current_output, vocab, with_eos=False)

        batch_logits, _ = model(feature)
        for batch_idx in range(batch_logits.shape[1]):
            logits = batch_logits[-1, batch_idx, :] / args.temperature
            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            if seq_i < args.min_seq_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        if current_output[batch_idx][-1] != eos_idx:
                            current_output[batch_idx].append(eos_idx)
                        print("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                if current_output[batch_idx][-1] != eos_idx:
                    current_output[batch_idx].append(eos_idx)
                continue
            if current_output[batch_idx][-1] != eos_idx:
                current_output[batch_idx].append(prev.item())

    current_output = [vs[1:] if vs[-1] == eos_idx else vs[1:] + [eos_idx]
                      for vs in current_output]
    pad_idx = vocab.stoi(PAD)
    padded = [vs + [pad_idx] * (args.max_seq_length+1 - len(vs))
              for vs in current_output]
    padded = torch.tensor(padded).T

    return current_output, padded


def build_input_from_segments(feature, current_output, vocab, with_eos=False):
    pad_idx = vocab.stoi(PAD)

    resp_pad = pad_sequence(list(map(torch.tensor, current_output)), 
            padding_value=pad_idx)
    resp_mask = generate_square_subsequent_mask(resp_pad.shape[0])
    resp_pad_mask = (resp_pad == pad_idx).T

    device = feature.context.device
    feature.resp = resp_pad.to(device)
    feature.resp_mask = resp_mask.to(device)
    feature.resp_pad_mask = resp_pad_mask.to(device)

    return feature

 
def create_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    log_fname = log_path + '/' + name + '.log'
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
     
