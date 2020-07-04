
import math
import random

import torch
import torch.nn as nn

import modules
from utils import mask_seq_batch


class PCCM(nn.Module):
    """Paper: Assigning personality/identity to a chatting machine for coherent conversation generation
    http://arxiv.org/abs/1706.02861
    """
    def __init__(
        self,
        post_encoder,
        profile_checker, 
        profile_detector,
        profile_emb,
        position_detector, 
        naive_forward_decoder,
        b_decoder,
        f_decoder,
        device
    ):
        super().__init__()

        self.post_encoder = post_encoder
        self.profile_checker = profile_checker
        self.profile_detector = profile_detector
        self.profile_emb = profile_emb
        self.position_detector = position_detector
        self.naive_forward_decoder = naive_forward_decoder
        self.b_decoder = b_decoder
        self.f_decoder = f_decoder
        self.device = device

    def forward(
        self,
        X,
        y, 
        y_lens,
        profiles,
        early_stage=False, 
        teacher_forcing_ratio=0.5
    ):
        ret = []
        max_len = y.shape[0]

        post_outs, post_hid = self.post_encoder(X)

        if early_stage:
            profile_exists = None
            beta = None
            v_pos = torch.tensor([random.randint(0, v-1) for v in y_lens]).to(self.device)
            profile_v = self.profile_emb(y[v_pos, torch.arange(y.shape[1])])
            no_profile_mask = torch.zeros_like(v_pos) == 0
            has_profile_mask = torch.ones_like(v_pos) == 1
            ret.append([None, None, None, None])
        else:
            # XXX: there maybe profile and not profile post in one batch,
            #      so the navie forward decoder and bidecoder 
            #      may process partial posts of one batch simultaneously
            profile_exists = self.profile_checker(post_outs)
            no_profile_mask = profile_exists <= 0.5
            has_profile_mask = profile_exists > 0.5
            no_profile = profile_exists[no_profile_mask]
            has_profile = profile_exists[has_profile_mask]
               
            outs = None
            if no_profile.shape[0] > 0:
                outs, _ = self.decode(
                        lambda out, hid, _: self.naive_forward_decoder(
                            out, hid, mask_seq_batch(post_outs, no_profile_mask)), 
                        0, max_len, mask_seq_batch(y, no_profile_mask), 
                        post_hid[no_profile_mask], teacher_forcing_ratio)
            ret.append([outs, None, None, None])

            beta = None
            v_pos = None
            if has_profile.shape[0] > 0:
                beta, j, profile_v = self.profile_detector(profiles, 
                        mask_seq_batch(post_outs, has_profile_mask))
                # batch_size
                v_pos = self.position_detector(mask_seq_batch(y, has_profile_mask), profile_v)

        b_outs = None
        f_outs = None
        if v_pos is not None:
            b_outs, b_rnn_outs = self.decode(
                    lambda out, hid, mask: self.b_decoder(out, hid, 
                        # mask_seq_batch(post_outs, has_profile_mask)[:, [i]], profile_v[[i]]), 
                        mask_seq_batch(post_outs, has_profile_mask)[:, mask], profile_v[mask]), 
                    v_pos, 0, mask_seq_batch(y, has_profile_mask), 
                    post_hid[has_profile_mask], teacher_forcing_ratio)
            f_outs, _ = self.decode(
                    lambda out, hid, mask: self.f_decoder(out, hid, 
                        mask_seq_batch(post_outs, has_profile_mask)[:, mask], 
                        profile_v[mask], b_rnn_outs.sum(dim=0)[mask]), 
                    v_pos, max_len, mask_seq_batch(y, has_profile_mask), 
                    post_hid[has_profile_mask], teacher_forcing_ratio)
        ret.append([f_outs, b_outs, beta, v_pos])

        return ret, profile_exists, no_profile_mask, has_profile_mask

    def _decode(
        self,
        decode_fn,
        start,
        end, 
        y,
        post_hid,
        teacher_forcing_ratio
    ):
        rnn_outs = None
        outs = torch.zeros(*y.shape[:2], self.f_decoder.output_dim).to(self.device)

        if type(start) == int and type(end) == int:
            hid = post_hid
            out = y[start]
            for t in range(start, end):
                out, hid, _ = decode_fn(out, hid, 0)
                outs[t] = out
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = out.max(1)[1]
                out = y[t] if teacher_force else top1
        else:
            rnn_outs = torch.zeros(*y.shape[:2], self.f_decoder.dec_hid_dim).to(self.device)
            # TODO: optimize
            # XXX: for every row in batch
            for i in range(0, y.shape[1]):
                s = start
                e = end
                hid = post_hid[[i]]
                if type(start) != int:
                    s = start[i].item()
                    out = y[s, [i]]
                else:
                    out = y[start, [i]]
                if type(end) != int:
                    e = end[i].item()

                if s > e:
                    e -= 1
                    s -= 1
                    rang = range(e, s, -1)
                else:
                    rang = range(s, e)
                for t in rang:
                    out, hid, rnn_out = decode_fn(out, hid, i)
                    outs[t, i] = out.squeeze(0)
                    rnn_outs[t, i] = rnn_out
                    teacher_force = random.random() < teacher_forcing_ratio
                    top1 = out.max(1)[1]
                    out = y[t, i].unsqueeze(0) if teacher_force else top1 

        return outs, rnn_outs

    def decode(
        self,
        decode_fn,
        start,
        end, 
        y,
        post_hid,
        teacher_forcing_ratio
    ):
        rnn_outs = None
        outs = torch.zeros(*y.shape[:2], 
                self.f_decoder.output_dim).to(self.device)

        if type(start) == int:
            hid = post_hid
            out = y[start]
            for t in range(start, end):
                out, hid, _ = decode_fn(out, hid, None)
                outs[t] = out
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = out.max(1)[1]
                out = y[t] if teacher_force else top1
        else:
            rnn_outs = torch.zeros(*y.shape[:2], 
                    self.f_decoder.dec_hid_dim).to(self.device)
                
            if end == 0:
                s = start.max()
                rg = range(s, -1, -1)
            else:
                s = start.min()
                rg = range(s, end)
            mask = start == s
            # remove this line
            # mask = start > -1

            hid = post_hid[mask]
            out = y[s, mask]
            top1 = None

            for t in rg:
                if top1 is not None:
                    old_mask = mask.clone()
                    add_mask = start == t
                    # XXX: this line same result, but backward failed
                    # mask |= add_mask
                    mask = mask | add_mask

                    if not teacher_force:
                        tmp = torch.zeros_like(old_mask).long()
                        tmp[old_mask] = top1
                        tmp[add_mask] = y[t, add_mask]
                        top1 = tmp[mask]
                    out = y[t, mask] if teacher_force else top1 
                    # out = y[t, mask]
                    # print(y.shape, out.shape)

                    tmp = torch.zeros(old_mask.shape[0], hid.shape[1]).to(self.device)
                    tmp[old_mask] = hid
                    tmp[add_mask] = post_hid[add_mask]
                    hid = tmp[mask]
                    # hid = post_hid[mask]

                out, hid, rnn_out = decode_fn(out, hid, mask)
                outs[t, mask] = out
                rnn_outs[t, mask] = rnn_out
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = out.max(1)[1]

        return outs, rnn_outs
 
                        

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            n = param.data.shape[-1]
            nn.init.uniform_(param.data, -math.sqrt(3/n), math.sqrt(3/n))
            # nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_word2vec(corpus_fname, vec_fname, vocab_fname):
    import gensim
    lss = gensim.models.word2vec.LineSentence(corpus_fname) 
    # skip-gram is more accuracy for most words, but CBOW is better for name similarity
    model = gensim.models.Word2Vec(lss)
    model.wv.save_word2vec_format(vec_fname, vocab_fname)


def load_embeddings_and_vocab(vec_fname, vocab_fname, truncate_freq=97):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_fname, vocab_fname)
    if truncate_freq is not None:
        vocab = {}
        vectors = []
        for k, v in model.vocab.items():
            if v.count >= truncate_freq:
                vocab[k] = v
        # vectors is ordered by count
        vectors = model.vectors[:len(vocab)] 
    return torch.tensor(vectors), vocab
