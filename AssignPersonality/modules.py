 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# use better encoder: 
# https://huggingface.co/transformers
class PostEncoder(nn.Module):
    def __init__(self,
        input_dim,
        emb_dim, 
        enc_hid_dim,
        dec_hid_dim, 
        num_layers,
        dropout,
        enc_bidi,
        emb_freeze, 
        pad_idx,
        embeddings=None
    ):
        super().__init__()
        self.enc_bidi = enc_bidi
        self.num_layers = num_layers
        self.num_directions = _num_dir(enc_bidi)
        self.enc_hid_dim = enc_hid_dim

        self.emb = embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.encoder = nn.GRU(emb_dim, enc_hid_dim, 
                num_layers=num_layers, bidirectional=enc_bidi)
        self.out = nn.Linear(enc_hid_dim * self.num_directions, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        emb = self.dropout(self.emb(X))
        # TODO: X are padded, use pack to optimize rnn encoder
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # https://www.kdnuggets.com/2018/06/taming-lstms-variable-sized-mini-batches-pytorch.html
        # https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        # https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html#define-encoder
        outs, hids = self.encoder(emb)

        # num_directions * batch_size * enc_hid_dim
        hid = hids.view(self.num_layers, 
                self.num_directions, *hids.shape[-2:])[-1]
        if self.enc_bidi:
            hid = torch.cat([hid[0], hid[1]], dim=1)
            outs = outs.view(*outs.shape[:2], self.num_directions, self.enc_hid_dim)
            outs = outs.sum(dim=2)
            # outs = outs.view(*outs.shape[:2], self.num_directions, self.enc_hid_dim)[:, :, -1, :]
            # outs = outs.squeeze(2)
        else:
            # batch_size * enc_hid_dim
            hid = hid[0]

        hid = torch.tanh(self.out(hid))

        return outs, hid


class ProfileChecker(nn.Module):
    def __init__(
        self,
        enc_hid_dim
    ):
        super().__init__()

        self.clf = nn.Linear(enc_hid_dim, 1)
 
    def forward(self, post_outs):
        post_outs_agg = agg_outs(post_outs)
        profile_exists = F.sigmoid(self.clf(post_outs_agg)).squeeze(1)
        return profile_exists

 
def agg_outs(outs, method='sum'):
    return outs.sum(dim=0)


class ProfileDetector(nn.Module):
    def __init__(
        self, 
        input_dim, 
        emb_dim, 
        enc_hid_dim, 
        n_profile, 
        dropout, 
        emb_freeze, 
        pad_idx,
        embeddings=None
    ):
        """use better classify: 
         https://github.com/brightmart/text_classification
         https://github.com/kk7nc/Text_Classification
         https://github.com/nadbordrozd/text-top-model
         https://github.com/prakashpandey9/Text-Classification-Pytorch
         https://github.com/dennybritz/cnn-text-classification-tf
        """
        super().__init__()
        self.emb_dim = emb_dim

        self.emb = embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.attn = nn.Linear(emb_dim*2 + enc_hid_dim, n_profile)
        # self.attn = nn.Linear(emb_dim + enc_hid_dim, n_profile)
        self.dropout = nn.Dropout(dropout)

    def forward(self, profiles, post_outs):
        post_outs_agg = agg_outs(post_outs)
        # profiles shape like: [k, v; k, v; k, v]
        # n_profile X 2 X emb_dim
        emb = self.dropout(self.emb(profiles))
        # emb = self.dropout(self.emb(profiles[:, 0]))
        # emb_v = self.dropout(self.emb(profiles[:, 1]))
        # emb_v = emb_v.unsqueeze(0).repeat(post_outs_agg.shape[0], 1, 1)
        # batch_size X n_profile X emb_dim * 2
        emb = emb.view(emb.shape[0], -1).unsqueeze(0).repeat(post_outs_agg.shape[0], 1, 1)

        # batch_size X n_profile X enc_hid_dim
        repeat_outs = post_outs_agg.unsqueeze(1).repeat(1, emb.shape[1], 1)
        energy = torch.tanh(self.attn(torch.cat([repeat_outs, emb], dim=2)))
        att = torch.sum(energy, dim=2)
        # batch_size X n_profile
        beta = F.log_softmax(att, dim=1)

        # batch_size
        j = beta.argmax(dim=1)
        # https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
        # profile_v = emb_v[:, j] (select columns) != below (select column per row)
        # batch_size X emb_dim
        profile_v = emb[torch.arange(emb.shape[0]), j].view(emb.shape[0], 2, self.emb_dim)[:, 1]
        # profile_v = emb_v[torch.arange(len(emb_v)), j]

        return beta, j, profile_v


class ProfileEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        dropout,
        emb_freeze, 
        pad_idx,
        embeddings=None
    ):
        super().__init__()

        self.emb = embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, profile):
        # seq_len X batch_size X emb_dim
        emb = self.dropout(self.emb(profile))
        return emb
 

class PositionDetector(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        dropout,
        emb_freeze, 
        pad_idx,
        embeddings=None
    ):
        super().__init__()

        self.emb = embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, profile_v):
        # seq_len X batch_size X emb_dim
        emb = self.dropout(self.emb(X))
        sim = F.cosine_similarity(emb, profile_v.unsqueeze(0), dim=2)
        v_pos = sim.argmax(dim=0)
        return v_pos

     
# use better decoder: 
# https://huggingface.co/transformers
class _BaseDecoder(nn.Module):
    def __init__(self,
        input_dim,
        emb_dim, 
        enc_hid_dim,
        dec_hid_dim, 
        attn_dim,
        num_layers,
        dropout, 
        attention,
        emb_freeze, 
        pad_idx,
        embeddings=None
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        dec_input_dim = self.calc_dec_input_dim()
        self.output_dim = input_dim
        self.num_layers = num_layers

        self.emb = embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.decoder = nn.GRU(dec_input_dim, dec_hid_dim, 
                num_layers=num_layers, bidirectional=False)
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, self.output_dim)
        self.attend = attention 
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X,
        hid, 
        post_outs,
        profile_v=None,
        b_rnn_outs=None
    ):
        a = self.attend(hid, post_outs)
        c = torch.bmm(a.unsqueeze(1), post_outs.permute(1, 0, 2))
        c = c.permute(1, 0, 2)

        X = X.unsqueeze(0)
        emb = self.dropout(self.emb(X))

        rnn_input = [emb, c]
        if profile_v is not None:
            rnn_input = [emb, profile_v.unsqueeze(0), c]
        if b_rnn_outs is not None:
            rnn_input = [emb, profile_v.unsqueeze(0), c, b_rnn_outs.unsqueeze(0)]
        outs, hids = self.decoder(torch.cat(rnn_input, dim=2), 
                hid.unsqueeze(0).repeat(self.num_layers, 1, 1))
        hid = hids.view(self.num_layers, 1, *hids.shape[-2:])[-1, 0]
        rnn_outs = outs.squeeze(0)

        emb = emb.squeeze(0)
        outs = outs.squeeze(0)
        c = c.squeeze(0)
        out = self.out(torch.cat([outs, c, emb], dim=1))

        return out, hid, rnn_outs

    def calc_dec_input_dim(self):
        raise Exception('Did not implemented.')

    def __attention(
        self,
        last_decoder_hid,
        post_hid
    ):
        a = F.softmax(self.attend(nn.concat([last_decoder_hid, post_hid])))
        c = a * post_hid
        return c 
 

class NaiveForwardDecoder(_BaseDecoder):
    def calc_dec_input_dim(self):
        return self.emb_dim + self.enc_hid_dim

    def forward(self, X, hid, post_outs):
        return super().forward(X, hid, post_outs)
         
 
class BiForwardDecoder(_BaseDecoder):
    def calc_dec_input_dim(self):
        return self.emb_dim + self.enc_hid_dim + self.emb_dim + self.dec_hid_dim

    def forward(
        self,
        X,
        hid,
        post_outs,
        profile_v,
        b_rnn_outs
    ):
        assert profile_v is not None
        assert b_rnn_outs is not None
        return super().forward(X, hid, post_outs, profile_v, b_rnn_outs)
         
 
class BiBackwardDecoder(_BaseDecoder):
    def calc_dec_input_dim(self):
        return self.emb_dim + self.enc_hid_dim + self.emb_dim

    def forward(
        self,
        X,
        hid,
        post_outs,
        profile_v,
    ):
        assert profile_v is not None
        return super().forward(X, hid, post_outs, profile_v)
               
                               
class Attention(nn.Module):
    """different attention implementions
     https://nbviewer.jupyter.org/github/susanli2016/NLP-with-Python/blob/master/Attention%20Basics.ipynb
     https://github.com/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)-Torch.py
     https://github.com/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi-LSTM(Attention)-Torch.py
     https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py#L318
     https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html#defining-our-nn-module-and-optimizer
     https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html#define-decoders-attention-module
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

    def forward(self, decoder_hid, post_outs):
        post_len = post_outs.shape[0]
        repeat_decoder_hid = decoder_hid.unsqueeze(1).repeat(1, post_len, 1)
        post_outs = post_outs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeat_decoder_hid,
            post_outs),
            dim=2)))
        att = torch.sum(energy, dim=2)

        return F.softmax(att, dim=1)


def _num_dir(enc_bidi):
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

