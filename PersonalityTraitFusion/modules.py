 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TraitEncoder(nn.Module):
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

    def forward(self, X):
        # n_trait X 2
        emb = self.dropout(self.emb(X))
        # n_trait X 2*emb_dim
        emb = emb.view(emb.shape[0], -1)

        return emb


class TraitFusion(nn.Module):
    def __init__(
        self,
        method,
        attention,
    ):
        methods = dict(
            attention=self.attention_method,
            average=self.average_method,
            concat=self.concat_method,
        )
        self.method_fn = methods[method]
        self.attention = attention

        super().__init__()

    def attention_method(self, prev_dec_hid, traits_enc):
        batch_size = prev_dec_hid.shape[0]
        traits_enc = traits_enc.unsqueeze(1).repeat(1, batch_size, 1)

        a = self.attend(prev_dec_hid, traits_enc)
        c = torch.bmm(a.unsqueeze(1), traits_enc.permute(1, 0, 2))
        c = c.permute(1, 0, 2)

        return c

    def average_method(self, prev_dec_hid, traits_enc): 
        batch_size = prev_dec_hid.shape[0]
        return traits_enc.mean(dim=0)\
                .unsqueeze(0).unsqueeze(0).repeat(1, batch_size, 1)

    def concat_method(self, prev_dec_hid, traits_enc):
        batch_size = prev_dec_hid.shape[0]
        return torch.cat(traits_enc, dim=1)\
                .unsqueeze(1).repeat(1, batch_size, 1)

    def forward(self, prev_dec_hid, traits_enc):
        return self.method_fn(prev_dec_hid, traits_enc)

 
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

     
class RespDecoder(nn.Module):
    def __init__(self,
        input_dim,
        emb_dim, 
        enc_hid_dim,
        dec_hid_dim, 
        attn_dim,
        num_layers,
        dropout, 
        attention,
        dec_method,
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
        self.dec_method = dec_method

        self.emb = embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.decoder = nn.GRU(dec_input_dim, dec_hid_dim, 
                num_layers=num_layers, bidirectional=False)
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, self.output_dim)
        self.bias_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, self.output_dim)
        self.Vo = nn.Parameter()
        self.attend = attention 
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X,
        hid, 
        post_outs,
        traits_fus=None
    ):
        if self.dec_method = 'PAA':
            a = self.attend(hid, post_outs, traits_fus)
        elif self.dec_method = 'PAB':
            a = self.attend(hid, post_outs)
        c = torch.bmm(a.unsqueeze(1), post_outs.permute(1, 0, 2))
        c = c.permute(1, 0, 2)

        X = X.unsqueeze(0)
        emb = self.dropout(self.emb(X))

        rnn_input = [emb, c]
        if traits_fus is not None:
            rnn_input = [emb, traits_fus.unsqueeze(0), c]
        outs, hids = self.decoder(torch.cat(rnn_input, dim=2), 
                hid.unsqueeze(0).repeat(self.num_layers, 1, 1))
        hid = hids.view(self.num_layers, 1, *hids.shape[-2:])[-1, 0]
        rnn_outs = outs.squeeze(0)

        emb = emb.squeeze(0)
        outs = outs.squeeze(0)
        c = c.squeeze(0)
        if self.dec_method = 'PAA':
            out = self.out(torch.cat([outs, c, emb], dim=1))
        elif self.dec_method = 'PAB':
            alpha = F.sigmoid(self.Vo.T.dot(hid))
            out = alpha * self.out(torch.cat([hid, c, emb], dim=1)) +  \
                    (1 - alpha) * self.bias_out(traits_fus)

        return out, hid, rnn_outs

                               
class Attention(nn.Module):
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

