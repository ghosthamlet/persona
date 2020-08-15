
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
    def __init__(
        self,
        context_emb,
        persona_emb,
        output_emb,
        post_encoder,
        resp_decoder,
        generater,
    ):
        super().__init__()

        self.context_emb = context_emb
        self.persona_emb = persona_emb
        self.output_emb = output_emb
        self.post_encoder = post_encoder
        self.resp_decoder = resp_decoder
        self.generater = generater

        self._share_emb()
        self._share_encoder_decoder()
        utils.xavier_init_weights(self)

    def forward(self, feature):
        context_enc, persona_enc = self.encode(feature)
        out = self.decode(feature, context_enc, persona_enc)

        return out

    def encode(self, feature):
        context_emb = self.context_emb(feature)
        persona_emb = self.persona_emb(feature.persona)

        context_enc = self.post_encoder(context_emb, feature.context_pad_mask)
        persona_enc = self.post_encoder(persona_emb, feature.persona_pad_mask)
 
        return context_enc, persona_enc

    def decode(self, feature, context_enc, persona_enc):
        resp_enc = self.output_emb(feature.resp)
        out = self.resp_decoder(resp_enc, context_enc, persona_enc, 
                memory_key_padding_mask=feature.context_pad_mask, 
                tgt_mask=feature.resp_mask, 
                tgt_key_padding_mask=feature.resp_pad_mask, 
                persona_pad_mask=feature.persona_pad_mask) 

        enc_lm = self.output_emb(feature.lm.x)
        out_lm = self.resp_decoder(enc_lm, tgt_mask=feature.lm.x_mask,
                tgt_key_padding_mask=feature.lm.x_pad_mask) 

        return self.generate(out), self.generate(out_lm)

    def generate(self, enc):
        return self.generater(enc)

    def predict(self, X):
        pass

    def _share_emb(self):
        self.context_emb.emb.weight = self.output_emb.emb.weight
        self.persona_emb.emb.weight = self.output_emb.emb.weight
        self.generater.out.weight = self.output_emb.emb.weight

    def _share_encoder_decoder(self):
        for i, layer in enumerate(self.post_encoder.transformer_encoder.layers):
            d_layer = self.resp_decoder.layers[i]
            layer.self_attn = d_layer.multihead_attn
            layer.linear1 = d_layer.linear1
            layer.linear2 = d_layer.linear2
            layer.norm1 = d_layer.norm1
            layer.norm2 = d_layer.norm2

    @staticmethod
    def build(
        args, 
        input_dim,
        output_dim,
        vocab,
        embeddings=None
    ):   
        pad_idx = vocab.stoi(utils.PAD)
        sep_idx = vocab.stoi(utils.SEP)
        spe1_idx = vocab.stoi(utils.SPE1)
        spe2_idx = vocab.stoi(utils.SPE2)

        context_emb = modules.ContextEmb(sep_idx, spe1_idx, spe2_idx,
                input_dim, args.d_model, args.emb_freeze,
                pad_idx, args.enc_dropout, embeddings)
        persona_emb = modules.PersonaEmb(input_dim, args.d_model, args.emb_freeze,
                pad_idx, embeddings)
        output_emb = modules.OutputEmb(input_dim, args.d_model, args.emb_freeze,
                pad_idx, args.enc_dropout, embeddings)

        post_encoder = modules.TransformerEncoder(input_dim, args.d_model, args.d_ff, 
                args.n_head, args.num_layers, args.enc_dropout)

        resp_decoder_layer = modules.TransformerDecoderLayer(args.d_model, args.n_head, 
                args.attn_alpha, args.d_ff, args.dec_dropout)
        resp_decoder = modules.TransformerDecoder(resp_decoder_layer, args.num_layers)
        generater = modules.Generater(args.d_model, output_dim)

        if args.n_epochs_early_stage > 0:
            model = LM(
                    context_emb, persona_emb, output_emb,
                    post_encoder, resp_decoder, generater)
        else:
            model = AR(
                    context_emb, persona_emb, output_emb,
                    post_encoder, resp_decoder, generater)

        return model
 
    @staticmethod
    def loss(loss_fn, out, out_lm, resp, lm_y):
        loss = loss_fn(out[:-1].view(-1, out.shape[-1]), resp[1:].view(-1))
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
       
