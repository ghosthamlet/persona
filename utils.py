
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

 
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

