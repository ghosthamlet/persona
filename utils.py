
import torch
import torch.nn as nn

                                
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

