
import math
import random

import torch
import torch.nn as nn

import modules
from utils import mask_seq_batch

 
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
 

class AR(nn.Module):
    def __init__(
        self,
        sep_idx,
        spe1_idx,
        spe2_idx,

        context_emb,
        persona_emb,
        output_emb,
        post_encoder,
        resp_decoder,
        generater,
    ):
        super().__init__()

        self.sep_idx = sep_idx
        self.spe1_idx = spe1_idx
        self.spe2_idx = spe2_idx

        self.context_emb = context_emb
        self.persona_emb = persona_emb
        self.output_emb = output_emb
        self.post_encoder = post_encoder
        self.resp_decoder = resp_decoder
        self.generater = generater

        self._share_encoder_decoder()
        self._init_weights()

    def forward(
        self,
        X,
        y, 
        persona,
        masks,
    ):
        src_mask, tgt_mask = masks
        context_enc, persona_enc = self.encode(X, persona, src_mask)
        out = self.decode(y, context_enc, persona_enc, tgt_mask, src_mask)

        return out

    def encode(self, X, persona, src_mask):
        context_emb = self.context_emb(X, 
                self.sep_idx, self.spe1_idx, self.spe2_idx)
        persona_emb = self.persona_emb(persona)

        context_enc = self.post_encoder(context_emb, src_mask)
        persona_enc = self.post_encoder(persona_emb)
 
        return context_enc, persona_enc

    def decode(self, y, context_enc, persona_enc, tgt_mask, src_mask):
        out = self.resp_decoder(y, context_enc, persona_enc, tgt_mask, src_mask) 
                # y_key_padding_mask, context_key_padding_mask)
        return self.generate(out)

    def generate(self, enc):
        return self.generater(enc)

    def predict(self, X):
        max_len = y.shape[0]

        post_outs, post_hid = self.encoder(X)
        trait_enc = self.trait_encoder(profiles)

        hid = post_hid
        out = y[start]
        rnn_outs = None
        outs = torch.zeros(*y.shape[:2], 
                self.resp_decoder.output_dim).to(X.device)
        for t in range(start, end):
            trait_fus = self.trait_fusion(hid, trait_enc)
            out, hid, _ = self.resp_decoder(out, hid, post_outs, trait_fus)
            outs[t] = out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out.max(1)[1]
            out = y[t] if teacher_force else top1

        return outs

    def _share_encoder_decoder(self):
        for i, layer in enumernate(self.post_encoder.transformer_encoder.layers):
            d_layers = self.resp_decoder.layers
            d_layers[i].self_attn = layer.self_attn
            d_layers[i].linear1 = layer.linear1
            d_layers[i].linear2 = layer.linear2
            d_layers[i].norm1 = layer.norm1
            d_layers[i].norm2 = layer.norm2

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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


def build_word2vec(corpus_fname, vec_fname, vocab_fname, max_vocab_size):
    import gensim
    lss = gensim.models.word2vec.LineSentence(corpus_fname) 
    # skip-gram is more accuracy for most words, but CBOW is better for name similarity
    model = gensim.models.Word2Vec(lss, max_final_vocab=max_vocab_size)
    model.wv.save_word2vec_format(vec_fname, vocab_fname)


def load_embeddings_and_vocab(vec_fname, vocab_fname):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_fname, vocab_fname)
    return torch.tensor(model.vectors), model.vocab
