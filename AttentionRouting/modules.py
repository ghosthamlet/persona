
import math
import sys
# for import parent utils
sys.path.append('../')
import utils
 
import torch
import torch.nn as nn
import torch.nn.functional as F


# FIXME: fix old code, all combined dim with view, the new dim should be dim0 * dim1, not dim0 + dim1
# FIXME: attention should all mask pad, see http://nlp.seas.harvard.edu/2018/04/03/attention.html, or softmax will add score to 0s


class ContextEmb(nn.Module):
    def __init__(
        self,
        sep_idx,
        spe1_idx,
        spe2_idx,

        input_dim,
        emb_dim, 
        emb_freeze, 
        pad_idx,
        dropout,
        embeddings=None 
    ):
        super().__init__()
        self.sep_idx = sep_idx
        self.spe1_idx = spe1_idx
        self.spe2_idx = spe2_idx
        self.emb_dim = emb_dim

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.pos_encoder = utils.PositionalEncoding(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature):
        # context: seq_len X batch_size
        #      seq: ..._SEP...
        # segs: seq_len X batch_size
        #      _SPE1 _SPE1 _SPE1 _SPE2 _SPE2 _SPE2
        # personas_no_tag: 2 X n_persona X batch_size
        # tags: 2 X n_tags X batch_size (n_tags has pad, as it is different in speakers)
        emb = self.emb(feature.context) * math.sqrt(self.emb_dim)

        if True:
            # XXX: paper no this
            segs_emb = self.emb(feature.segs)

            # 2 X n_persona X batch_size X emb_dim
            personas_emb = self.emb(feature.personas_no_tag)
            # 2 X n_tags X batch_size X emb_dim
            tags_emb = self.emb(feature.tags)
            # 2 X batch_size X emb_dim
            personas_emb = torch.cat([personas_emb, tags_emb], dim=1).sum(dim=1)
            # segs spe1_idx and spe2_idx is not a must
            # (segs == idx) can be created from iterate context
            fn = lambda emb, idx, i: torch.where(
                   (feature.segs == idx).unsqueeze(2).repeat(1, 1, emb.shape[2]), 
                   emb + personas_emb[i], emb)
            emb = fn(emb, self.spe1_idx, 0)
            emb = fn(emb, self.spe2_idx, 1)

        emb = emb + self.pos_encoder(emb)
        emb = self.dropout(emb)

        return emb


class PersonaEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        emb_freeze, 
        pad_idx,
        embeddings=None 
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)

    def forward(self, persona):
        # seq_len(k;v) X batch_size X emb_dim
        emb = self.emb(persona) * math.sqrt(self.emb_dim)

        return emb                

 
class OutputEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        emb_freeze, 
        pad_idx,
        dropout,
        embeddings=None 
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.pos_encoder = utils.PositionalEncoding(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output):
        emb = self.emb(output) * math.sqrt(self.emb_dim)
        emb = self.dropout(emb + self.pos_encoder(emb))

        return emb                
 
 
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        n_hid,
        n_head,
        num_layers,
        dropout
    ):
        super().__init__()
        self.num_layers = num_layers

        encoder_layers = TransformerEncoderLayer(emb_dim, n_head, n_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, emb, mask=None):
        outs = self.transformer_encoder(emb, src_key_padding_mask=mask)

        return outs


class TransformerEncoderLayer(nn.Module):
    r"""Derived from torch.nn.TransformerEncoderLayer
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.norm2(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    r"""Derived from torch.nn.TransformerDecoderLayer
    """

    def __init__(self, d_model, nhead, attn_alpha,
            dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_alpha = attn_alpha
        self.cls = nn.Linear(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def forward(self, tgt, memory, persona,
            tgt_mask=None, memory_mask=None,
            tgt_key_padding_mask=None, memory_key_padding_mask=None,
            persona_pad_mask=None):
        tgt = self.norm2(tgt)

        if True:
            attn_t = 0
            attn_c = 0
            attn_prev = self.multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            if persona is not None and memory is not None:
                attn_t = self.multihead_attn(tgt, persona, persona, 
                        key_padding_mask=persona_pad_mask)[0]
                attn_c = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, 
                        key_padding_mask=memory_key_padding_mask)[0]
            alpha = self.cls(memory) if self.attn_alpha is None else self.attn_alpha 
            attn_merge = alpha*attn_t + (1-alpha)*attn_c + attn_c + attn_prev
            attn_merge = tgt + self.dropout(attn_merge)
            attn_merge = self.norm1(attn_merge)
        else:
            attn_prev = self.multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            attn_prev = tgt + self.dropout1(attn_prev)
            attn_prev = self.norm1(attn_prev)
            attn_t = self.multihead_attn(attn_prev, persona, persona, 
                    key_padding_mask=persona_pad_mask)[0]
            attn_c = self.multihead_attn(attn_prev, memory, memory, attn_mask=memory_mask, 
                    key_padding_mask=memory_key_padding_mask)[0]
            alpha = self.cls(memory) if self.attn_alpha is None else self.attn_alpha 
            attn_merge = alpha*attn_t + (1-alpha)*attn_c + attn_c
            attn_merge = attn_prev + self.dropout(attn_merge)
            attn_merge = self.norm1(attn_merge)
 
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(attn_merge))))
        tgt = attn_merge + self.dropout2(tgt2)

        return tgt, alpha


class TransformerDecoder(nn.Module):
    r"""Derived from torch.nn.TransformerDecoder
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory=None, persona=None, 
            memory_mask=None, memory_key_padding_mask=None,
            tgt_mask=None, tgt_key_padding_mask=None,
            persona_pad_mask=None):
        """Train language model When memory and persona is None"""
        output = tgt

        for i in range(self.num_layers):
            output, alpha = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask, persona=persona,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    persona_pad_mask=persona_pad_mask)

        if self.norm:
            output = self.norm(output)

        return output


class Generater(nn.Module):
    def __init__(
        self,
        d_model,
        output_dim
    ):
        super().__init__()

        self.out = nn.Linear(d_model, output_dim, bias=False)

    def forward(self, enc):
        return self.out(enc)
