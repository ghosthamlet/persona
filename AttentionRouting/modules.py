
import ..utils
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextEmb(nn.Module):
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

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, sep_idx, spe1_idx, spe2_idx):
        # utt: seq_len X batch_size
        #      seq: ..._SEP...
        # segs: seq_len X batch_size
        #      _SPE1 _SPE1 _SPE1 _SPE2 _SPE2 _SPE2
        # persona: 2 X n_persona
        # tags: 2 X n_tags (list type, as n_tag is different in speakers)
        utt, segs, persona, tags = context
        emb = self.emb(utt)
        emb = emb + self.pos_encoder(emb)

        # XXX: paper no this
        segs_emb = self.emb(segs)

        # 2 X n_persona X emb_dim
        persona_emb = self.emb(persona)
        # 2 X 1 X emb_dim
        tags_emb = torch.cat([
            self.emb(tags[0]).mean(dim=0, keepdim=True),
            self.emb(tags[1]).mean(dim=0, keepdim=True)
            ], dim=0).unsqueeze(1)
        # 2 X n_persona+1 X emb_dim --> 2 X emb_dim
        persona_emb = torch.cat([persona_emb, tags_emb], dim=1).sum(dim=1)
        # segs spe1_idx and spe2_idx is not a must
        # (segs == idx) can be created from iterate utt
        fn = lambda idx, i: torch.where(
                (segs == idx).unsqueeze(2).repeat(1, 1, emb.shape[2]), 
                emb + persona_emb[i], emb)
        emb = fn(spe1_idx, 0)
        emb = fn(spe2_idx, 1)

        emb = self.dropout(emb)

        return emb


class PersonaEmb(nn.Module):
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

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, persona):
        # persona: pack all k v into a word seq
        # seq_len X emb_dim
        emb = self.dropout(self.emb(persona))
        # seq_len X batch_size X emb_dim
        emb = emb.unsqueeze(1)

        return emb                

 
class OutputEmb(nn.Module):
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

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output):
        emb = self.dropout(self.emb(output))
        emb = emb + self.pos_encoder(emb)

        return emb                

 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

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

 
class Encoder(nn.Module):
    def __init__(self,
        input_dim,
        emb_dim, 
        enc_hid_dim,
        dec_hid_dim, 
        num_layers,
        dropout,
    ):
        super().__init__()
        self.enc_bidi = enc_bidi
        self.num_layers = num_layers
        self.enc_hid_dim = enc_hid_dim

        encoder_layers = nn.TransformerEncoderLayer(input_dim, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        # TODO: X are padded, use pack to optimize rnn encoder
        outs = self.transformer_encoder(emb)

        return outs


class TransformerSharedLayer(Module):
    r"""Dirived from torch.nn.TransformerDecoderLayer
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.cls = nn.Linear(d_model, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, persona, 
            tgt_mask=None, tgt_key_padding_mask=None):
       #tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
       #                      key_padding_mask=tgt_key_padding_mask)[0]
       #tgt = tgt + self.dropout1(tgt2)
       #tgt = self.norm1(tgt)
       #tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
       #                           key_padding_mask=memory_key_padding_mask)[0]
       #tgt = tgt + self.dropout2(tgt2)
       #tgt = self.norm2(tgt)

        attn_t = self.multihead_attn(tgt, persona, persona)[0]
        attn_c = self.multihead_attn(tgt, memory, memory)[0]
        attn_prev = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        alpha = self.cls(memory)
        attn_merge = tgt + self.dropout2(alpha*attn_t + (1-alpha)*attn_c + attn_c + attn_prev)
        attn_merge = self.norm2(attn_merge)
 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(attn_merge))))
        tgt = attn_merge + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, alpha


class TransformerShared(Module):
    r"""Dirived from torch.nn.TransformerDecoder
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, profiles, 
            tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


