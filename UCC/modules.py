
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
        d_model,
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
        self.proj = nn.Linear(emb_dim, d_model)
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
        emb = self.proj(emb)
        emb = self.dropout(emb)

        return emb


class PersonaEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        emb_freeze, 
        d_model,
        pad_idx,
        embeddings=None 
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.proj = nn.Linear(emb_dim, d_model)

    def forward(self, persona):
        # seq_len(k;v) X batch_size X emb_dim
        emb = self.emb(persona) * math.sqrt(self.emb_dim)
        emb = self.proj(emb)

        return emb                

 
class OutputEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        emb_freeze, 
        d_model,
        pad_idx,
        dropout,
        embeddings=None 
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.pos_encoder = utils.PositionalEncoding(emb_dim)
        self.proj = nn.Linear(emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output):
        emb = self.emb(output) * math.sqrt(self.emb_dim)
        emb = emb + self.pos_encoder(emb)
        emb = self.proj(emb)
        emb = self.dropout(emb)

        return emb                
 
 
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        n_hid,
        n_head,
        num_layers,
        dropout,
        attn_act,
        adapter_finetune, 
        adapter_d_ff,
    ):
        super().__init__()
        self.num_layers = num_layers

        use_rezero = True
        if use_rezero:
            encoder_layers = RZTXEncoderLayer(emb_dim, n_head, n_hid, dropout,
                   attn_act, adapter_finetune, adapter_d_ff)
        else:
            encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_head, n_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, emb, mask=None):
        outs = self.transformer_encoder(emb, src_key_padding_mask=mask)

        return outs


class TransformerDecoderLayer(nn.Module):
    r"""Derived from torch.nn.TransformerDecoderLayer
    """

    def __init__(self, d_model, nhead, attn_alpha,
            dim_feedforward=2048, dropout=0.1, activation="relu",
            adapter_finetune=False, adapter_d_ff=2048):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.factor_ff = True
        if self.factor_ff:
            in_ff = int(dim_feedforward/4)
           #self.linear1 = nn.Linear(d_model, in_ff)
           #self.fac_linear1 = nn.Linear(in_ff, in_ff)
           #self.fac_linear2 = nn.Linear(in_ff, in_ff)
           #self.linear2 = nn.Linear(in_ff, d_model)
            self.linear1 = nn.Linear(d_model, 100)
            self.fac_linear1 = nn.Linear(100, dim_feedforward)
            self.fac_linear2 = nn.Linear(dim_feedforward, 100)
            self.linear2 = nn.Linear(100, d_model)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_alpha = attn_alpha
        if self.attn_alpha is None:
            self.cls = nn.Linear(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.resweight = nn.Parameter(torch.Tensor([0]))
 
        self.adapter_finetune = adapter_finetune
        if self.adapter_finetune:
            self.ada_linear1 = Linear(d_model, adapter_d_ff)
            self.ada_dropout1 = Dropout(dropout)
            self.ada_linear2 = Linear(adapter_d_ff, d_model)
            self.ada_dropout2 = Dropout(dropout)

            self.multihead_attn.requires_grad_(False)
            #self.multihead_attn.in_proj_weight.requires_grad = False
            #self.multihead_attn.in_proj_bias.requires_grad = False
            self.linear1.requires_grad_(False)
            self.linear2.requires_grad_(False)
            #self.linear1.weight.requires_grad = False
            #self.linear1.bias.requires_grad = False
            #self.linear2.weight.requires_grad = False
            #self.linear2.bias.requires_grad = False
            #self.resweight.requires_grad = False

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def forward(self, tgt, memory, persona,
            tgt_mask=None, memory_mask=None,
            tgt_key_padding_mask=None, memory_key_padding_mask=None,
            persona_pad_mask=None):
       #tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
       #                      key_padding_mask=tgt_key_padding_mask)[0]
       #tgt = tgt + self.dropout1(tgt2)
       #tgt = self.norm1(tgt)
       #tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
       #                           key_padding_mask=memory_key_padding_mask)[0]
       #tgt = tgt + self.dropout2(tgt2)
       #tgt = self.norm2(tgt)

        use_rezero = True
        if use_rezero:
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
            if persona is not None and memory is not None:
                attn_merge = tgt + 0.1*attn_t + 0.1*attn_c + self.dropout(attn_merge) * self.resweight
            else:
                attn_merge = tgt + self.dropout(attn_merge) * self.resweight

            if self.factor_ff:
               #tgt2 = self.fac_linear1(self.dropout1(self.activation(self.linear1(attn_merge))))
               #tgt = attn_merge + self.dropout2(tgt2) * self.resweight
               #tgt2 = self.linear2(self.dropout1(self.activation(self.fac_linear2(tgt))))
               #tgt = tgt + self.dropout2(tgt2) * self.resweight
                tgt2 = self.dropout1(self.fac_linear1(self.activation(self.linear1(attn_merge))))
                tgt2 = self.linear2(self.dropout1(self.activation(self.fac_linear2(tgt2))))
                tgt = attn_merge + self.dropout2(tgt2) * self.resweight
            else:
                tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(attn_merge))))
                tgt = attn_merge + self.dropout2(tgt2) * self.resweight
 
            if self.adapter_finetune:
                src2 = tgt            
                src2 = self.ada_linear2(self.ada_dropout1(self.activation(self.ada_linear1(src2))))
                src2 = src2 * self.resweight
                src = tgt + self.ada_dropout2(src2)
        elif True:
            # must start with small lr 1.5e-4
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
            if persona is not None and memory is not None:
                attn_merge = tgt + 0.1*attn_t + 0.1*attn_c + self.dropout(attn_merge)
            else:
                attn_merge = tgt + self.dropout(attn_merge)
            attn_merge = self.norm1(attn_merge)
 
            tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(attn_merge))))
            tgt = attn_merge + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        else:
            # this can start with large lr 0.05
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
            tgt = self.norm2(tgt)

        return tgt, alpha

 
# adapt from https://github.com/majumderb/rezero
# https://nbviewer.jupyter.org/github/tbachlechner/ReZero-examples/blob/master/ReZero-Deep_Fast_Transformer.ipynb
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear

class RZTXEncoderLayer(Module):
    r"""RZTXEncoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, d_model, nhead, 
            dim_feedforward=2048, dropout=0.1, activation='relu', 
            adapter_finetune=False, adapter_d_ff=2048):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.factor_ff = True
        if self.factor_ff:
            in_ff = int(dim_feedforward/4)
           #self.linear1 = nn.Linear(d_model, in_ff)
           #self.fac_linear1 = nn.Linear(in_ff, in_ff)
           #self.fac_linear2 = nn.Linear(in_ff, in_ff)
           #self.linear2 = nn.Linear(in_ff, d_model)
            self.linear1 = nn.Linear(d_model, 100)
            self.fac_linear1 = nn.Linear(100, dim_feedforward)
            self.fac_linear2 = nn.Linear(dim_feedforward, 100)
            self.linear2 = nn.Linear(100, d_model)
        else:
            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        self.adapter_finetune = adapter_finetune
        if self.adapter_finetune:
            self.ada_linear1 = Linear(d_model, adapter_d_ff)
            self.ada_dropout1 = Dropout(dropout)
            self.ada_linear2 = Linear(adapter_d_ff, d_model)
            self.ada_dropout2 = Dropout(dropout)

            self.self_attn.requires_grad_(False)
            #self.self_attn.in_proj_weight.requires_grad = False
            #self.self_attn.in_proj_bias.requires_grad = False
            self.linear1.requires_grad_(False)
            self.linear2.requires_grad_(False)
            #self.linear1.weight.requires_grad = False
            #self.linear1.bias.requires_grad = False
            #self.linear2.weight.requires_grad = False
            #self.linear2.bias.requires_grad = False
            #self.resweight.requires_grad = False
             
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        # Self attention layer
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0] # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        if self.factor_ff:
           #src2 = self.fac_linear1(self.dropout(self.activation(self.linear1(src))))
           #src = src + self.dropout2(src2 * self.resweight)
           #src2 = self.linear2(self.dropout(self.activation(self.fac_linear2(src))))
           #src = src + self.dropout2(src2 * self.resweight)
            src2 = self.dropout1(self.fac_linear1(self.activation(self.linear1(src))))
            src2 = self.linear2(self.dropout(self.activation(self.fac_linear2(src2))))
            src = src + self.dropout2(src2 * self.resweight)
        else:
            src2 = src            
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src2 = src2 * self.resweight
            src = src + self.dropout2(src2)

        if self.adapter_finetune:
            src2 = src            
            src2 = self.ada_linear2(self.ada_dropout1(self.activation(self.ada_linear1(src2))))
            src2 = src2 * self.resweight
            src = src + self.ada_dropout2(src2)
 
        return src

class RZTXDecoderLayer(nn.Module):
    r"""RZTXDecoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> decoder_layer = RZTXDecoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = decoder_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) * self.resweight
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2) * self.resweight

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2) * self.resweight
        return tgt


class TransformerDecoder(nn.Module):
    r"""Derived from torch.nn.TransformerDecoder
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory=None, profiles=None, 
            memory_mask=None, memory_key_padding_mask=None,
            tgt_mask=None, tgt_key_padding_mask=None,
            persona_pad_mask=None):
        """Train language model When memory and profiles is None"""
        output = tgt

        for i in range(self.num_layers):
            output, alpha = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask, persona=profiles,
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

