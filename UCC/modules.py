
import math
import random
import sys
# for import parent utils
sys.path.append('../')
import utils
 
import torch
import torch.nn as nn
import torch.nn.functional as F


# FIXME: fix old code, all combined dim with view, the new dim should be dim0 * dim1, not dim0 + dim1
# FIXME: attention should all mask pad, see http://nlp.seas.harvard.edu/2018/04/03/attention.html, or softmax will add score to 0s


_ALL_HID_STATES_IDX = -1


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
        persona_vocab_size,
        persona_emb_dim,
        embeddings=None,
        pretrain_feature_model=None,
    ):
        super().__init__()
        self.sep_idx = sep_idx
        self.spe1_idx = spe1_idx
        self.spe2_idx = spe2_idx
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.d_model = d_model

        self.pretrain_feature = pretrain_feature_model is not None
        if self.pretrain_feature:
            self.pos_encoder = utils.PositionalEncoding(emb_dim*2)
            self.proj = nn.Linear(emb_dim*2, d_model)
        else:
            self.pos_encoder = utils.PositionalEncoding(emb_dim)
            # self.pos_encoder = nn.Embedding(512, emb_dim)
            self.proj = nn.Linear(emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if self.pretrain_feature:
            self.emb1 = pretrain_feature_model
        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)
        self.persona_emb = self.emb
        if persona_emb_dim is not None:
            self.persona_emb = nn.Embedding(persona_vocab_size, persona_emb_dim)

    def forward(self, feature):
        def orig_emb(feature):
            # context: seq_len X batch_size
            #      seq: ..._SEP...
            # segs: seq_len X batch_size
            #      _SPE1 _SPE1 _SPE1 _SPE2 _SPE2 _SPE2
            # personas_no_tag: 2 X n_persona X batch_size
            # tags: 2 X n_tags X batch_size (n_tags has pad, as it is different in speakers)
            emb = self.emb(feature.context) * math.sqrt(self.emb_dim)

            # segs_emb = self.emb(feature.segs)

            # 2 X n_persona X batch_size X emb_dim
            personas_emb = self.persona_emb(feature.personas_no_tag)
            # 2 X n_tags X batch_size X emb_dim
            tags_emb = self.persona_emb(feature.tags)
            # 2 X batch_size X emb_dim
            personas_emb = torch.cat([personas_emb, tags_emb], dim=1).sum(dim=1)
            # segs spe1_idx and spe2_idx is not a must
            # (segs == idx) can be created from iterate context
            fn = lambda emb, idx, i: torch.where(
                   (feature.segs == idx).unsqueeze(2).repeat(1, 1, emb.shape[2]), 
                   emb + personas_emb[i], emb)
            emb = fn(emb, self.spe1_idx, 0)
            emb = fn(emb, self.spe2_idx, 1)

            return emb 

        if self.pretrain_feature:
            def new_emb(feature):
                context = feature.context.transpose(0, 1)
                segs = feature.segs.transpose(0, 1)
                personas_no_tag = feature.personas_no_tag.permute(2, 0, 1)
                tags = feature.tags.permute(2, 0, 1)
                personas_no_tag = personas_no_tag.view(personas_no_tag.shape[0], -1)
                tags = tags.view(tags.shape[0], -1)
                context_pad_mask = (feature.context_pad_mask != 1).float()
                personas_no_tag_pad_mask = (feature.personas_no_tag_pad_mask.view(
                        feature.personas_no_tag_pad_mask.shape[0], -1) != 1).float()
                tags_pad_mask = (feature.tags_pad_mask.view(
                        feature.tags_pad_mask.shape[0], -1) != 1).float()

                personas_no_tag_position_ids = torch.zeros_like(personas_no_tag)
                tags_position_ids = torch.zeros_like(tags)
                persona_dim = 1

                emb = self.emb1(context, attention_mask=context_pad_mask)[_ALL_HID_STATES_IDX][-2]

                # segs_emb = self.emb1(segs)

                # batch_size X 2 * n_persona X emb_dim
                personas_emb = self.emb1(personas_no_tag, 
                        position_ids=personas_no_tag_position_ids,
                        attention_mask=personas_no_tag_pad_mask)[_ALL_HID_STATES_IDX][-2]
                # batch_size X 2 * n_tags X emb_dim
                tags_emb = self.emb1(tags,
                        position_ids=tags_position_ids,
                        attention_mask=tags_pad_mask)[_ALL_HID_STATES_IDX][-2]

                personas_emb = personas_emb.view(personas_emb.shape[0], 2, 
                        -1, personas_emb.shape[2])
                tags_emb = tags_emb.view(tags_emb.shape[0], 2, -1, tags_emb.shape[2])
                segs = segs.transpose(0, 1)
                personas_emb = personas_emb.permute(1, 2, 0, 3)
                tags_emb = tags_emb.permute(1, 2, 0, 3)
                emb = emb.transpose(0, 1)

                # 2 X batch_size X emb_dim
                personas_emb = torch.cat([personas_emb, tags_emb], dim=persona_dim).sum(dim=persona_dim)
                # segs spe1_idx and spe2_idx is not a must
                # (segs == idx) can be created from iterate context
                fn = lambda emb, idx, i: torch.where(
                       (segs == idx).unsqueeze(2).repeat(1, 1, emb.shape[2]), 
                       emb + personas_emb[i], emb)
                emb = fn(emb, self.spe1_idx, 0)
                emb = fn(emb, self.spe2_idx, 1)

                return emb

           #emb = torch.cat([new_emb(feature), orig_emb(feature)], dim=2)
           #emb = emb + self.pos_encoder(emb)
           #emb = self.proj(emb)
           #emb = self.dropout(emb)
            emb = new_emb(feature)
        else:
            emb = orig_emb(feature)
            emb = emb + self.pos_encoder(emb)
            # emb = emb + self.pos_encoder(create_position_ids(feature.context))
            emb = self.proj(emb)
            emb = self.dropout(emb)

        return emb


def create_position_ids(input_ids):
    device = input_ids.device
    input_shape = input_ids.shape
    seq_length = input_shape[0]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(1).expand(input_shape)
    return position_ids


class PersonaEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        emb_freeze, 
        d_model,
        pad_idx,
        dropout,
        embeddings=None,
        pretrain_feature_model=None,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.pretrain_feature = pretrain_feature_model is not None
        if self.pretrain_feature:
            self.proj = nn.Linear(emb_dim*2, d_model)
        else:
            self.proj = nn.Linear(emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if self.pretrain_feature:
            self.emb1 = pretrain_feature_model
        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)

    def forward(self, persona, persona_pad_mask):
        def orig_emb(persona):
            # seq_len(k;v) X batch_size X emb_dim
            emb = self.emb(persona) * math.sqrt(self.emb_dim)
            return emb 

        if self.pretrain_feature:
            def new_emb(persona, persona_pad_mask):
                persona = persona.transpose(0, 1)
                persona_position_ids = torch.zeros_like(persona)
                persona_pad_mask = (persona_pad_mask != 1).float()

                # batch_size X seq_len(k;v) X emb_dim
                emb = self.emb1(persona, 
                        position_ids=persona_position_ids,
                        attention_mask=persona_pad_mask)[_ALL_HID_STATES_IDX][-2]

                emb = emb.transpose(0, 1)

                return emb

           #emb = torch.cat([new_emb(persona, persona_pad_mask), orig_emb(persona)], dim=2)
           #emb = self.proj(emb)
           #emb = self.dropout(emb)
            emb = new_emb(persona, persona_pad_mask)
        else:
            emb = orig_emb(persona)
            emb = self.proj(emb)
            emb = self.dropout(emb)

        return emb                


class SeqEmb(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim, 
        emb_freeze, 
        d_model,
        pad_idx,
        dropout,
        embeddings=None,
        pretrain_feature_model=None,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.pretrain_feature = pretrain_feature_model is not None
        if self.pretrain_feature:
            self.pos_encoder = utils.PositionalEncoding(emb_dim*2)
            self.proj = nn.Linear(emb_dim*2, d_model)
        else:
            self.pos_encoder = utils.PositionalEncoding(emb_dim)
            self.proj = nn.Linear(emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if self.pretrain_feature:
            self.emb1 = pretrain_feature_model
        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)

    def forward(self, x, x_pad_mask):
        def orig_emb(x):
            emb = self.emb(x) * math.sqrt(self.emb_dim)
            return emb 

        if self.pretrain_feature:
            def new_emb(x, x_pad_mask):
                x = x.transpose(0, 1)
                x_pad_mask = (x_pad_mask != 1).float()

                emb = self.emb1(x, 
                        attention_mask=x_pad_mask)[_ALL_HID_STATES_IDX][-2]

                emb = emb.transpose(0, 1)

                return emb

            # emb = torch.cat([new_emb(x, x_pad_mask), orig_emb(x)], dim=2)
            emb = new_emb(x, x_pad_mask)
        else:
            emb = orig_emb(x)
            emb = emb + self.pos_encoder(emb)
            emb = self.proj(emb)
            emb = self.dropout(emb)

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
        embeddings=None,
        pretrain_feature_model=None,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.pretrain_feature = pretrain_feature_model is not None
        if self.pretrain_feature:
            self.pos_encoder = utils.PositionalEncoding(emb_dim*2)
            self.proj = nn.Linear(emb_dim*2, d_model)
        else:
            self.pos_encoder = utils.PositionalEncoding(emb_dim)
            self.proj = nn.Linear(emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if self.pretrain_feature:
            self.emb1 = pretrain_feature_model
        self.emb = utils.embedding(input_dim, emb_dim, embeddings, emb_freeze, pad_idx)

    def forward(self, output, output_pad_mask):
        def orig_emb(x):
            emb = self.emb(output) * math.sqrt(self.emb_dim)
            return emb 

        if self.pretrain_feature:
            def new_emb(x, x_pad_mask):
                x = x.transpose(0, 1)
                x_pad_mask = (x_pad_mask != 1).float()

                # XXX: must get pretrain_feature_model emb, not hidden state after self attention
                #      or future output token will be attended
                emb = self.emb1(x, 
                        attention_mask=x_pad_mask)[_ALL_HID_STATES_IDX][0]

                emb = emb.transpose(0, 1)

                return emb

            # emb = torch.cat([new_emb(output, output_pad_mask), orig_emb(output)], dim=2)
            emb = new_emb(output, output_pad_mask)
        else:
            emb = orig_emb(output)
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
        num_groups,
        dropout,
        attn_act,
        factor_ff,
        adapter_finetune, 
        adapter_d_ff,
        use_rezero=True,
        norm=None
    ):
        super().__init__()
        self.num_layers = num_layers

        # TODO: move out layer define
        if use_rezero:
            encoder_layer = RZTXEncoderLayer(emb_dim, n_head, n_hid, dropout,
                   attn_act, factor_ff, adapter_finetune, adapter_d_ff)
        else:
            encoder_layer = TransformerEncoderLayer(emb_dim, n_head, n_hid, dropout)

        layers_in_group = num_layers // num_groups
        self.num_groups = num_groups
        self.layers = nn.modules.transformer._get_clones(encoder_layer, layers_in_group)
        self.norm = norm

    def forward(self, src_emb, pad_mask=None):
        output = src_emb

        layers_in_group = len(self.layers)
        for _ in range(self.num_groups):
            for i in range(layers_in_group):
                output = self.layers[i](output, src_mask=None,
                                        src_key_padding_mask=pad_mask)

        if self.norm:
            output = self.norm(output)

        return output

 
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
            dim_feedforward=2048, dropout=0.1, activation="relu",
            factor_ff=False, adapter_finetune=False, adapter_d_ff=2048,
            use_rezero=True, auxiliary_task='MLM'):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.factor_ff = factor_ff
        self.use_rezero = use_rezero
        self.auxiliary_task = auxiliary_task

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

        self.pre_norm = nn.LayerNorm(d_model)
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
        tgt = self.pre_norm(tgt)

        if self.use_rezero:
            attn_t = 0
            attn_c = 0
            attn_prev = self.multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            if persona is not None and memory is not None:
                attn_t = self.multihead_attn(tgt, persona, persona, 
                        key_padding_mask=persona_pad_mask)[0]
                attn_c = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, 
                        key_padding_mask=memory_key_padding_mask)[0]
            elif memory is not None:
                # for mlm
                attn_c = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, 
                        key_padding_mask=memory_key_padding_mask)[0]
            alpha = self.cls(memory) if self.attn_alpha is None else self.attn_alpha 
            attn_merge = alpha*attn_t + (1-alpha)*attn_c + attn_c + attn_prev
            a = 0.1
            if persona is not None and memory is not None:
                attn_merge = tgt + a*attn_t + a*attn_c + self.dropout(attn_merge) * self.resweight
            elif memory is not None:
                attn_merge = tgt + a*attn_c + self.dropout(attn_merge) * self.resweight
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
        elif self.auxiliary_task == 'MLM':
            attn_t = 0
            attn_c = 0
            attn_prev = self.multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            if persona is not None and memory is not None:
                attn_t = self.multihead_attn(tgt, persona, persona, 
                        key_padding_mask=persona_pad_mask)[0]
                attn_c = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, 
                        key_padding_mask=memory_key_padding_mask)[0]
            elif memory is not None:
                attn_c = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, 
                        key_padding_mask=memory_key_padding_mask)[0]

            alpha = self.cls(memory) if self.attn_alpha is None else self.attn_alpha 
            attn_merge = alpha*attn_t + (1-alpha)*attn_c + attn_c + attn_prev
            a = 0.
            if persona is not None and memory is not None:
                attn_merge = tgt + a*attn_t + a*attn_c + self.dropout(attn_merge)
            elif memory is not None:
                attn_merge = tgt + a*attn_c + self.dropout(attn_merge)
            attn_merge = self.norm1(attn_merge)
 
            if self.factor_ff:
                tgt2 = self.dropout1(self.fac_linear1(self.activation(self.linear1(attn_merge))))
                tgt2 = self.linear2(self.dropout1(self.activation(self.fac_linear2(tgt2))))
                tgt = attn_merge + self.dropout2(tgt2)
            else:
                tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(attn_merge))))
                tgt = attn_merge + self.dropout2(tgt2)
        elif True:
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

            if self.factor_ff:
                tgt2 = self.dropout1(self.fac_linear1(self.activation(self.linear1(attn_merge))))
                tgt2 = self.linear2(self.dropout1(self.activation(self.fac_linear2(tgt2))))
                tgt = attn_merge + self.dropout2(tgt2)
            else:
                tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(attn_merge))))
                tgt = attn_merge + self.dropout2(tgt2) 
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

        return tgt, alpha

 
# adapt from https://github.com/majumderb/rezero
# https://nbviewer.jupyter.org/github/tbachlechner/ReZero-examples/blob/master/ReZero-Deep_Fast_Transformer.ipynb
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.modules.module import Module
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
            factor_ff=False, adapter_finetune=False, adapter_d_ff=2048):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.factor_ff = factor_ff
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
        self.pre_norm = nn.LayerNorm(d_model)

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
        src = self.pre_norm(src)

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

    def __init__(self, decoder_layer, num_layers, num_groups, norm=None):
        super(TransformerDecoder, self).__init__()

        layers_in_group = num_layers // num_groups
        self.num_groups = num_groups
        self.layers = nn.modules.transformer._get_clones(decoder_layer, layers_in_group)
        self.norm = norm

    def forward(self, tgt_emb, memory=None, persona=None, 
            memory_mask=None, memory_key_padding_mask=None,
            tgt_mask=None, tgt_key_padding_mask=None,
            persona_pad_mask=None):
        """Train language model When memory and persona is None"""
        output = tgt_emb

        layers_in_group = len(self.layers)
        for _ in range(self.num_groups):
            for i in range(layers_in_group):
                output, alpha = self.layers[i](
                        output, memory, persona=persona,
                        tgt_mask=tgt_mask, memory_mask=memory_mask, 
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        persona_pad_mask=persona_pad_mask)

        if self.norm:
            output = self.norm(output)

        return output


class Generater(nn.Module):
    def __init__(
        self,
        emb_dim,
        d_model,
        output_dim
    ):
        super().__init__()

        self.out = nn.Linear(d_model, output_dim, bias=False)

    def forward(self, enc):
        return self.out(enc)

 
class _MemInput(nn.Module):
    def __init__(
        self,
        emb,
        encoder,
    ):
        super().__init__()

        self.emb = emb
        self.encoder = encoder
 
    def forward(self, persona, post_query, persona_pad_mask):
        emb = self.emb(persona, persona_pad_mask)
        emb = self.encoder(emb, persona_pad_mask)
        e = emb.transpose(0, 1).bmm(post_query.sum(dim=0).unsqueeze(2))
        mask = persona_pad_mask.float().masked_fill(
                persona_pad_mask == 1, float('-inf')).unsqueeze(2)
        e = e + mask
        p = F.softmax(e, dim=1).transpose(0, 1)

        return p
       
         
class MemInput(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.emb = nn.Embedding(vocab_size, emb_dim)
 
    def forward(self, persona, post_query, persona_pad_mask):
        emb = self.emb(persona)

        if len(post_query.shape) == 2:
            # sentence emb
            post_query = post_query.unsqueeze(2)
        else:
            seq_len = post_query.shape[0]
            pos = torch.arange(1, seq_len+1).unsqueeze(1).to(post_query.device)
            k = torch.arange(1, self.emb_dim+1).to(post_query.device)
            pos = (1 - pos / seq_len) - (k / self.emb_dim) * (1 - 2 * pos / seq_len)
            post_query = post_query * pos.unsqueeze(1)

            post_query = post_query.sum(dim=0).unsqueeze(2)

        e = emb.transpose(0, 1).bmm(post_query)
        mask = persona_pad_mask.float().masked_fill(
                persona_pad_mask == 1, float('-inf')).unsqueeze(2)
        e = e + mask
        p = F.softmax(e, dim=1).transpose(0, 1)

        return p
         

class _MemOutput(nn.Module):
    def __init__(
        self,
        emb,
        encoder,
    ):
        super().__init__()

        self.emb = emb
        self.encoder = encoder
 
    def forward(self, persona, persona_input_mem, persona_pad_mask):
        emb = self.emb(persona, persona_pad_mask)
        emb = self.encoder(emb, persona_pad_mask)

        return emb.permute(1, 2, 0).bmm(
               persona_input_mem.transpose(0, 1)).permute(2, 0, 1)
 
 
class MemOutput(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, persona, persona_input_mem, persona_pad_mask):
        emb = self.emb(persona)

        return emb.permute(1, 2, 0).bmm(
               persona_input_mem.transpose(0, 1)).permute(2, 0, 1)
 
