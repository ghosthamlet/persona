
import copy
import math
import random

import torch
import torch.nn as nn

import sys
# for import parent utils
sys.path.append('../')
import utils

import modules


class AR(nn.Module):
    """
    Much simpler transformer implemention:
    https://github.com/atselousov/transformer_chatbot/blob/agent/model/transformer_module.py
    """
    def __init__(
        self,
        context_emb,
        persona_emb,
        seq_emb,
        output_emb,
        post_encoder,
        resp_decoder,
        generater,
        factor_ff,
        adapter_finetune,
        share_encoder_decoder,
        pretrain_feature_model,
        pretrain_feature_type,
        persona_vocab_size,
        use_mem_n2n,
        mem_n2n_hops,
        mem_n2n_layer_share,
        auxiliary_task=None,
    ):
        super().__init__()

        self.context_emb = context_emb
        self.persona_emb = persona_emb
        self.seq_emb = seq_emb
        self.output_emb = output_emb
        self.post_encoder = post_encoder
        self.resp_decoder = resp_decoder
        self.generater = generater
        self.adapter_finetune = adapter_finetune
        self.factor_ff = factor_ff
        self.auxiliary_task = auxiliary_task
        self.share_encoder_decoder = share_encoder_decoder
        self.pretrain_feature_type = pretrain_feature_type
        self.use_mem_n2n = use_mem_n2n
        self.mem_n2n_hops = mem_n2n_hops
        self.mem_n2n_layer_share = mem_n2n_layer_share

        if self.use_mem_n2n:
            if self.mem_n2n_layer_share == 'adjacent':
                self.mem_input = nn.modules.transformer._get_clones(
                        modules.MemInput(persona_vocab_size, context_emb.d_model),
                        self.mem_n2n_hops)
                self.mem_output = nn.modules.transformer._get_clones(
                        modules.MemOutput(persona_vocab_size, context_emb.d_model),
                        self.mem_n2n_hops)
                self._share_mem_n2n_layers()
            else:
                self.mem_output_map = nn.Linear(context_emb.d_model, context_emb.d_model, bias=False)
                #self.mem_input = modules.MemInput(copy.deepcopy(self.seq_emb),
                #        copy.deepcopy(self.post_encoder))
                self.mem_input = modules.MemInput(persona_vocab_size, context_emb.d_model)
                self.mem_output = modules.MemOutput(persona_vocab_size, context_emb.d_model)

        self._share_emb()
        if self.share_encoder_decoder:
            self._share_encoder_decoder()
        utils.xavier_init_weights(self)

        self.pretrain_feature_model = None
        if pretrain_feature_type == 'mem_n2n':
            self.pretrain_feature_model = pretrain_feature_model

        if pretrain_feature_model is not None and 'weight' in pretrain_feature_type:
            self._init_with_pretrain_feature_model_emb(pretrain_feature_model)

            if pretrain_feature_model.base_model_prefix != 'albert':
                self._init_with_pretrain_feature_model_last_layer(pretrain_feature_model)
            else:
                self._init_with_pretrain_feature_model_layers(pretrain_feature_model)

    def forward(self, feature):
        if self.use_mem_n2n:
            context_enc, persona_enc, x_mlm_enc = self.mem_n2n(feature)
        else:
            context_enc, persona_enc, x_mlm_enc = self.encode(feature)
        out = self.decode(feature, context_enc, persona_enc, x_mlm_enc)

        return out

    def encode(self, feature):
        context_emb = self.context_emb(feature)
        persona_emb = self.persona_emb(feature.persona, feature.persona_pad_mask)

        context_enc = self.post_encoder(context_emb, feature.context_pad_mask)
        persona_enc = self.post_encoder(persona_emb, feature.persona_pad_mask)

        x_mlm_enc = self.encode_lm(feature)
 
        return context_enc, persona_enc, x_mlm_enc

    def mem_n2n(self, feature):
        persona_enc = None
        if self.pretrain_feature_model is None:
            post_emb = self.seq_emb(feature.post, feature.post_pad_mask)
        else:
            post_emb = self.pretrain_feature_model(feature.post.T, 
                    attention_mask=feature.post_pad_mask)[0][:, 0]
        # worse
        # post_enc = self.post_encoder(post_emb, feature.post_pad_mask)
        context_emb = self.context_emb(feature)

        last_post_emb = post_emb
        for i in range(self.mem_n2n_hops):
            if self.mem_n2n_layer_share == 'adjacent':
                p = self.mem_input[i](feature.persona, last_post_emb, feature.persona_pad_mask)
                o = self.mem_output[i](feature.persona, p, feature.persona_pad_mask)
                last_post_emb = last_post_emb + o
            else:
                p = self.mem_input(feature.persona, last_post_emb, feature.persona_pad_mask)
                o = self.mem_output(feature.persona, p, feature.persona_pad_mask)
                last_post_emb = self.mem_output_map(last_post_emb) + o

        context_emb = context_emb + o
        context_enc = self.post_encoder(context_emb, feature.context_pad_mask)

        x_mlm_enc = self.encode_lm(feature)
        
        return context_enc, persona_enc, x_mlm_enc

    def _share_mem_n2n_layers(self):
        for i in range(self.mem_n2n_hops):
            if i < self.mem_n2n_hops-1:
                self.mem_output[i].emb.weight = self.mem_input[i+1].emb.weight

    def encode_lm(self, feature):
        x_mlm_enc = None
        if self.auxiliary_task == 'MLM':
            x_mlm_emb = self.seq_emb(feature.lm.x_mlm, feature.lm.x_mlm_pad_mask)
            x_mlm_enc = self.post_encoder(x_mlm_emb, feature.lm.x_mlm_pad_mask)
        return x_mlm_enc

    def decode(self, feature, context_enc, persona_enc, x_mlm_enc):
        persona_bias = None
       #post_emb = self.seq_emb(feature.post, feature.post_pad_mask)
       #p = self.mem_input(feature.persona, post_emb, feature.persona_pad_mask)
       #persona_bias = self.mem_output(feature.persona, p, feature.persona_pad_mask)
       #context_enc = context_enc + persona_bias

        resp_emb = self.output_emb(feature.resp, feature.resp_pad_mask)
        out = self.resp_decoder(
                resp_emb, memory=context_enc, persona=persona_enc, 
                memory_key_padding_mask=feature.context_pad_mask, 
                tgt_mask=feature.resp_mask, 
                tgt_key_padding_mask=feature.resp_pad_mask, 
                persona_pad_mask=feature.persona_pad_mask) 


        if persona_bias is not None:
            out_gen = self.generate(out + persona_bias)
        else:
            out_gen = self.generate(out)

        enc_lm = self.output_emb(feature.lm.x, 
                feature.lm.x_pad_mask)
        out_lm = self.resp_decoder(enc_lm, memory=x_mlm_enc, 
                memory_key_padding_mask=feature.lm.x_mlm_pad_mask,
                tgt_mask=feature.lm.x_mask,
                tgt_key_padding_mask=feature.lm.x_pad_mask) 
        out_lm_gen = self.generate(out_lm)

        return out_gen, out_lm_gen

    def generate(self, enc):
        return self.generater(enc)

    def predict(self, X):
        pass

    def _share_emb(self):
        self.context_emb.emb.weight = self.output_emb.emb.weight
        #self.context_emb.proj.weight = self.output_emb.proj.weight
        if self.use_mem_n2n:
            self.context_emb.persona_emb.weight = self.persona_emb.emb.weight
        else:
            self.persona_emb.emb.weight = self.output_emb.emb.weight
        #self.persona_emb.proj.weight = self.output_emb.proj.weight
        self.seq_emb.emb.weight = self.output_emb.emb.weight

    def _init_with_pretrain_feature_model_emb(self, pretrain_feature_model):
        m = pretrain_feature_model.embeddings
        self.output_emb.emb.weight = copy.deepcopy(m.word_embeddings.weight)
        self.output_emb.emb.weight.requires_grad = True
        self._share_emb()

    def _share_encoder_decoder(self):
        for i, layer in enumerate(self.post_encoder.layers):
            d_layer = self.resp_decoder.layers[i]
            layer.self_attn = d_layer.multihead_attn
            layer.linear1 = d_layer.linear1
            layer.linear2 = d_layer.linear2
            layer.norm1 = d_layer.norm1
            layer.norm2 = d_layer.norm2
            layer.resweight = d_layer.resweight
            layer.pre_norm = d_layer.pre_norm

            if self.factor_ff:
                layer.fac_linear1 = d_layer.fac_linear1
                layer.fac_linear2 = d_layer.fac_linear2

            if self.adapter_finetune:
                layer.ada_linear1 = d_layer.ada_linear1
                layer.ada_linear2 = d_layer.ada_linear2

    def _init_with_pretrain_feature_model_last_layer(self, pretrain_feature_model):
        def fn(from_layer, to_layer):
            m = pretrain_feature_model.encoder.layer[from_layer]
            att = m.attention
            att_self = att.self
            att_output = att.output.dense
            linear1 = m.intermediate.dense
            linear2 = m.output.dense
            norm1 = att.output.LayerNorm
            norm2 = m.output.LayerNorm

            layer0 = self.resp_decoder.layers[to_layer]
            layer0.multihead_attn.in_proj_weight = nn.Parameter(torch.cat([
                copy.deepcopy(att_self.query.weight), 
                copy.deepcopy(att_self.key.weight), 
                copy.deepcopy(att_self.value.weight)
                ], dim=0))
            layer0.multihead_attn.in_proj_bias = nn.Parameter(torch.cat([
                copy.deepcopy(att_self.query.bias), 
                copy.deepcopy(att_self.key.bias), 
                copy.deepcopy(att_self.value.bias)
                ], dim=0))                           
            layer0.multihead_attn.out_proj = copy.deepcopy(att_output)

            layer0.linear1 = copy.deepcopy(linear1)
            layer0.linear2 = copy.deepcopy(linear2)
            layer0.norm1 = copy.deepcopy(norm1)
            layer0.norm2 = copy.deepcopy(norm2)

            layer0.requires_grad_(True)

        l = len(self.resp_decoder.layers)
        for i in range(l):
            fn(-(l-i), i)

        if self.share_encoder_decoder:
            self._share_encoder_decoder()

    def _init_with_pretrain_feature_model_layers(self, pretrain_feature_model):
        m = pretrain_feature_model.encoder.albert_layer_groups[-1].albert_layers[-1]
        att = m.attention
        att_self = att
        att_output = att.dense
        linear1 = m.ffn
        linear2 = m.ffn_output
        norm1 = att.LayerNorm
        norm2 = m.full_layer_layer_norm

        if self.share_encoder_decoder:
            layer0 = self.resp_decoder.layers[0]
            multihead_attn = layer0.multihead_attn
        else:
            layer0 = self.post_encoder.layers[0]
            multihead_attn = layer0.self_attn

        multihead_attn.in_proj_weight = nn.Parameter(torch.cat([
            copy.deepcopy(att_self.query.weight), 
            copy.deepcopy(att_self.key.weight), 
            copy.deepcopy(att_self.value.weight)
            ], dim=0))
        multihead_attn.in_proj_bias = nn.Parameter(torch.cat([
            copy.deepcopy(att_self.query.bias), 
            copy.deepcopy(att_self.key.bias), 
            copy.deepcopy(att_self.value.bias)
            ], dim=0))                           
        multihead_attn.out_proj = copy.deepcopy(att_output)

        layer0.linear1 = copy.deepcopy(linear1)
        layer0.linear2 = copy.deepcopy(linear2)
        layer0.norm1 = copy.deepcopy(norm1)
        layer0.norm2 = copy.deepcopy(norm2)

        layer0.requires_grad_(True)

        if self.share_encoder_decoder:
            self._share_encoder_decoder()

    @staticmethod
    def build(
        args, 
        input_dim,
        output_dim,
        vocab,
        embeddings=None,
        pretrain_feature_model=None,
    ):
        pad_idx = vocab.stoi(utils.PAD)
        sep_idx = vocab.stoi(utils.SEP)
        spe1_idx = vocab.stoi(utils.SPE1)
        spe2_idx = vocab.stoi(utils.SPE2)

        fn = None
        _pretrain_feature_model = pretrain_feature_model
        if pretrain_feature_model is not None and args.pretrain_feature_type != 'weight':
            # don't define as layer, or the model weights will be saved to checkpoint
            def fn(x, position_ids=None, attention_mask=None):
                # 'requires_grad_(False)' just for disable backward calc grad, 
                # add 'with torch.no_grad()' to disable save activation
                with torch.no_grad():
                    return pretrain_feature_model(
                            x, position_ids=position_ids,
                            attention_mask=attention_mask)
                            # last layer hid
                            #attention_mask=attention_mask)[0]
                            # last layer CLS hid
                            # attention_mask=attention_mask)[0][0].unsqueeze(0)
                            # emb
                            #attention_mask=attention_mask)[1][0]
            if args.pretrain_feature_type == 'mem_n2n':
                _pretrain_feature_model = fn
                fn = None

        context_emb = modules.ContextEmb(sep_idx, spe1_idx, spe2_idx,
                input_dim, args.emb_dim, args.emb_freeze, 
                args.d_model, pad_idx, args.dropout, 
                args.persona_vocab_size, args.use_mem_n2n, 
                embeddings, fn)
        if args.use_mem_n2n:
            persona_emb = modules.PersonaEmb(
                    args.persona_vocab_size, args.emb_dim, False,
                    args.d_model, pad_idx, args.dropout, None, fn)
        else:
            persona_emb = modules.PersonaEmb(input_dim, args.emb_dim, args.emb_freeze,
                    args.d_model, pad_idx, args.dropout, embeddings, fn)
        seq_emb = modules.SeqEmb(input_dim, args.emb_dim, args.emb_freeze,
                args.d_model, pad_idx, args.dropout, embeddings, fn)
        output_emb = modules.OutputEmb(input_dim, args.emb_dim, args.emb_freeze,
                args.d_model, pad_idx, args.dropout, embeddings, fn)

        post_encoder = modules.TransformerEncoder(input_dim, args.d_model, args.d_ff, 
                args.n_head, args.num_layers, args.num_groups, args.dropout,
                'relu', args.factor_ff, args.adapter_finetune, args.adapter_d_ff,
                args.use_rezero)

        resp_decoder_layer = modules.TransformerDecoderLayer(args.d_model, args.n_head, 
                args.attn_alpha, args.d_ff, args.dropout, 
                'relu', args.factor_ff, args.adapter_finetune, args.adapter_d_ff,
                args.use_rezero)
        resp_decoder = modules.TransformerDecoder(resp_decoder_layer, args.num_layers, args.num_groups)
        generater = modules.Generater(args.emb_dim, args.d_model, output_dim)

        if args.n_epochs_early_stage > 0:
            model = LM(
                    context_emb, persona_emb, seq_emb, 
                    output_emb, post_encoder, resp_decoder, 
                    generater, args.factor_ff, args.adapter_finetune, 
                    args.share_encoder_decoder, _pretrain_feature_model, 
                    args.pretrain_feature_type, args.persona_vocab_size, 
                    args.use_mem_n2n, args.mem_n2n_hops, args,mem_n2n_layer_share
                    )
        else:
            model = AR(
                    context_emb, persona_emb, seq_emb, 
                    output_emb, post_encoder, resp_decoder, 
                    generater, args.factor_ff, args.adapter_finetune, 
                    args.share_encoder_decoder, _pretrain_feature_model, 
                    args.pretrain_feature_type, args.persona_vocab_size, 
                    args.use_mem_n2n, args.mem_n2n_hops, args.mem_n2n_layer_share,
                    args.auxiliary_task
                    )

        return model

    @staticmethod
    def loss(auxiliary_task, loss_fn, out, out_lm, resp, lm_y):
        loss = loss_fn(out[:-1].view(-1, out.shape[-1]), resp[1:].view(-1))
        loss_lm = None
        if auxiliary_task is not None:
            loss_lm = loss_fn(out_lm.view(-1, out.shape[-1]), lm_y.view(-1))

        return loss, loss_lm


class LM(AR):
    def forward(self, feature):
        enc = self.output_emb(feature.x)
        out = self.resp_decoder(enc, tgt_mask=feature.x_mask) 
        return self.generater(out)
 

class _LM(nn.Module):
    def __init__(
        self,
        output_emb,
        decoder,
        generater,
    ):
        super().__init__()

        self.output_emb = output_emb
        self.decoder = decoder
        self.generater = generater

        self._share_emb()
        utils.xavier_init_weights(self)

    def forward(self, feature):
        enc = self.output_emb(feature.x)
        out = self.decoder(enc, tgt_mask=feature.x_mask) 
        return self.generater(out)

    def _share_emb(self):
        self.generater.out.weight = self.output_emb.emb.weight
       

