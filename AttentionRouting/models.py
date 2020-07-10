
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
        post_encoder,
        resp_decoder,
    ):
        super().__init__()

        self.post_encoder = post_encoder
        self.resp_decoder = resp_decoder
        self.transformer = nn.Transformer()

    def forward(
        self,
        X,
        y, 
        profiles,
        teacher_forcing_ratio=0.5
    ):
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
