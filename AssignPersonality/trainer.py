
import os
import random
import math
import time
import argparse
import yaml
from collections import OrderedDict

import modules
import datasets
import models

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, profiles):
        args = self.parse_args()
        self.args = args
        self.best_valid_loss = float('inf')
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        self.set_random_seed()

        self.ensure_deps()

        if args.pretrain_emb and (
                not os.path.exists(args.vec_fname) 
                or not os.path.exists(args.vocab_fname)
        ):
            # XXX: datasets.retokenize all the chinese old datasets first 
            # 现成的已分词数据分词质量不好，造成word2vec效果下降，如小提琴的most_similar完全错误
            # TODO: try token to letter or subwords
            print('Pretraining word2vec...')
            models.build_word2vec(args.corpus_fname, args.vec_fname, args.vocab_fname)

        embeddings, gensim_vocab = None, None
        if args.pretrain_emb:
            print('Loading word2vec...')
            embeddings, gensim_vocab = models.load_embeddings_and_vocab(args.vec_fname, args.vocab_fname)
            embeddings = embeddings.to(self.device)
        self.vocab = datasets.Vocab(gensim_vocab, profiles, args.data_path)
        self.input_dim = len(self.vocab)
        # if special_tokens did not include in pretrain_emb, append zeros, disable emb_freeze
        if args.pretrain_emb:
            elen = embeddings.shape[0]
            if self.input_dim > elen:
                args.emb_freeze = False
                append = torch.zeros(self.input_dim - elen, embeddings.shape[1]).to(self.device)
                embeddings = torch.cat([embeddings, append], dim=0)

        self.profiles_features = torch.tensor(
                datasets.convert_profiles_to_features(self.vocab, profiles)).to(self.device)
        self.pad_idx = self.vocab.stoi(datasets.PAD)

        print('Build dataloaders...')
        self.build_dataloaders()
        print('Build model...')
        self.build_model(embeddings)
        print('Build loss fns...')
        self.build_loss_fns()

        print(f'The model has {models.count_parameters(self.model):,} trainable parameters')

    def set_random_seed(self):
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if self.device == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
         
    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_file', default='configs/default.yaml', type=str, required=False, 
            help='Provide config in config_file or as other commandline args')
        parser.add_argument('--device', default='cuda', type=str, required=False, help='Use cpu for easy debug')
        parser.add_argument('--seed', default=42, type=int, required=False, help='')
        parser.add_argument('--n_epochs', default=10, type=int, required=False, help='')
        parser.add_argument('--n_epochs_early_stage', default=0, type=int, required=False, help='')
        parser.add_argument('--clip_grad', default=1, type=int, required=False, help='')
        parser.add_argument('--batch_size', default=128, type=int, required=False, help='')
        parser.add_argument('--limit_example_length', default=256, type=int, required=False, help='')
        parser.add_argument('--max_seq_length', default=300, type=int, required=False, help='')
        parser.add_argument('--shuffle_data', action='store_true', required=False, help='')
        parser.add_argument('--pretrain_emb', action='store_true', required=False, help='')

        parser.add_argument('--emb_freeze', action='store_true', required=False, help='')
        parser.add_argument('--enc_bidi', action='store_true', required=False, help='')
        parser.add_argument('--enc_num_layers', default=4, type=int, required=False, help='')
        parser.add_argument('--dec_num_layers', default=4, type=int, required=False, help='')
        parser.add_argument('--enc_emb_dim', default=100, type=int, required=False, help='')
        parser.add_argument('--dec_emb_dim', default=100, type=int, required=False, help='')
        parser.add_argument('--enc_hid_dim', default=64, type=int, required=False, help='')
        parser.add_argument('--dec_hid_dim', default=64, type=int, required=False, help='')
        parser.add_argument('--attn_dim', default=8, type=int, required=False, help='')
        parser.add_argument('--enc_dropout', default=0.5, type=float, required=False, help='')
        parser.add_argument('--dec_dropout', default=0.5, type=float, required=False, help='')
        parser.add_argument('--enc_rnn_dropout', default=0, type=float, required=False, help='')
        parser.add_argument('--dec_rnn_dropout', default=0, type=float, required=False, help='')

        parser.add_argument('--alpha', default=1, type=float, required=False, help='weight of second loss')
        parser.add_argument('--lr', default=0.5, type=float, required=False, help='')
        parser.add_argument('--weight_decay', default=0.99, type=float, required=False, help='')

        parser.add_argument('--model_path', default='models/', type=str, required=False, help='')
        parser.add_argument('--pretrained_path', type=str, required=False, help='')
        parser.add_argument('--data_path', default='datas/', type=str, required=False, help='')
        parser.add_argument('--cache_path', default='caches/', type=str, required=False, help='')
        parser.add_argument('--corpus_fname', default='datas/corpus.txt', type=str, required=False, help='')
        parser.add_argument('--vec_fname', default='models/vec.txt', type=str, required=False, help='')
        parser.add_argument('--vocab_fname', default='models/vocab.txt', type=str, required=False, help='')

        args = parser.parse_args()
        if args.config_file != '':
            parser.set_defaults(**yaml.load(open(args.config_file)))
            args = parser.parse_args()

        return args

    def ensure_deps(self):
        if self.args.pretrain_emb:
            try:
                v = '3.8.3'
                import gensim
                assert gensim.__version__ >= v
            except:
                raise Exception('If pretrain_emb enabled, please install gensim>=%s' % v)
                                                                                         
    def build_dataloaders(self):
        args = self.args
        gb = lambda batch: datasets.generate_batch(batch, self.pad_idx)

        if args.n_epochs_early_stage > 0:
            ds = datasets.PersonaDataset(
                    self.vocab, args.max_seq_length, 
                    data_path=args.data_path, cache_path=args.cache_path, 
                    limit_length=args.limit_example_length, mode='early_stage_train')
            self.early_stage_train_iter = DataLoader(ds, batch_size=args.batch_size, 
                    collate_fn=gb, shuffle=args.shuffle_data) 

        ds = datasets.PersonaDataset(
                self.vocab, args.max_seq_length, 
                data_path=args.data_path, cache_path=args.cache_path, 
                limit_length=args.limit_example_length, mode='train')
        self.train_iter = DataLoader(ds, batch_size=args.batch_size, 
                collate_fn=gb, shuffle=args.shuffle_data) 

       #ds = datasets.PersonaDataset(
       #        self.vocab, args.max_seq_length, 
       #        data_path=args.data_path, cache_path=args.cache_path, 
       #        limit_length=args.limit_example_length, mode='valid')
       #self.valid_iter = DataLoader(ds, batch_size=args.batch_size,
       #        collate_fn=gb, shuffle=args.shuffle_data) 

       #ds = datasets.PersonaDataset(
       #        self.vocab, args.max_seq_length, 
       #        data_path=args.data_path, cache_path=args.cache_path, 
       #        limit_length=args.limit_example_length, mode='test')
       #self.test_iter = DataLoader(ds, batch_size=args.batch_size,
       #        collate_fn=gb, shuffle=args.shuffle_data)

    def build_model(self, embeddings):
        args = self.args
        output_dim = self.input_dim
        input_dim = self.input_dim
        pad_idx = self.pad_idx

        emb = modules.embedding(input_dim, args.enc_emb_dim, embeddings, args.emb_freeze, pad_idx)
        post_encoder = modules.PostEncoder(input_dim, args.enc_emb_dim, args.enc_hid_dim,
                args.dec_hid_dim, args.enc_num_layers, args.enc_dropout, args.enc_bidi,
                args.emb_freeze, pad_idx, embeddings
                )
        profile_checker = modules.ProfileChecker(args.enc_hid_dim)
        profile_detector = modules.ProfileDetector(
                input_dim, args.enc_emb_dim, args.enc_hid_dim,
                len(self.profiles_features), args.enc_dropout,
                args.emb_freeze, pad_idx, embeddings
                )
        profile_emb = modules.ProfileEmb(input_dim, args.enc_emb_dim, 
                args.enc_dropout, args.emb_freeze, pad_idx, embeddings
                )
        position_detector = modules.PositionDetector(input_dim, args.enc_emb_dim, 
                args.enc_dropout, args.emb_freeze, pad_idx, embeddings
                )

        attention = modules.Attention(args.enc_hid_dim, args.dec_hid_dim, 
                args.attn_dim)
        naive_forward_decoder = modules.NaiveForwardDecoder(
                input_dim, args.dec_emb_dim, args.enc_hid_dim,
                args.dec_hid_dim, args.attn_dim, args.dec_num_layers,
                args.dec_dropout, attention,
                args.emb_freeze, pad_idx, embeddings
                )
        attention = modules.Attention(args.enc_hid_dim, args.dec_hid_dim, 
                args.attn_dim)
        b_decoder = modules.BiBackwardDecoder(
                input_dim, args.dec_emb_dim, args.enc_hid_dim,
                args.dec_hid_dim, args.attn_dim, args.dec_num_layers,
                args.dec_dropout, attention,
                args.emb_freeze, pad_idx, embeddings
                )
        attention = modules.Attention(args.enc_hid_dim, args.dec_hid_dim, 
                args.attn_dim)
        f_decoder = modules.BiForwardDecoder(
                input_dim, args.dec_emb_dim, args.enc_hid_dim,
                args.dec_hid_dim, args.attn_dim, args.dec_num_layers,
                args.dec_dropout, attention,
                args.emb_freeze, pad_idx, embeddings
                )                                     

        self.model = models.PCCM(
                post_encoder, profile_checker, profile_detector,
                profile_emb, position_detector, naive_forward_decoder, 
                b_decoder, f_decoder, device=self.device
                ).to(self.device)
        # XXX: Adam is not good
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)

        print(self.model)

        if args.pretrained_path is None:
            self.model.apply(models.init_weights)
        else:
            self.load_model()

    def build_loss_fns(self):
        self.profile_exists_loss_fn = nn.NLLLoss(ignore_index=self.pad_idx)
        self.profile_detector_loss_fn = lambda input, target: -(input * target).sum() / input.shape[0]
        self.out_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def run_early_stage(self):
        for epoch in range(self.args.n_epochs_early_stage):
            start_time = time.time()

            train_loss = self.train(early_stage=True, data_iter=self.early_stage_train_iter)
            self.save_model(epoch)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    def run(self):
        if self.args.n_epochs_early_stage > 0:
            print('Run early stage...')
            trainer.run_early_stage()

        print('Run main stage...')

        for epoch in range(self.args.n_epochs):
            start_time = time.time()

            train_loss = self.train(early_stage=False)
            self.save_model(epoch)

            valid_loss = self.eval()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        test_loss = self.eval(self.test_iter)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    def train(self, data_iter=None, early_stage=False):
        self.model.train()

        if data_iter is None:
            data_iter = self.train_iter

        epoch_loss = 0
        for _, (X, y, X_lens, y_lens, profile_key) in enumerate(data_iter):
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)
            profile_y = profile_key[:, 0].float().to(self.device)
            profile_j = profile_key[:, 1].float().to(self.device)

            decoder_outs, profile_exists, no_profile_mask, has_profile_mask = self.model(
                    X, y, y_lens, self.profiles_features, early_stage)
            (out, *_), (f_out, b_out, beta, v_pos) = decoder_outs
            exists_no_profile = out is not None
            exists_has_profile = f_out is not None

            if exists_has_profile:
                y_pos = y[:, has_profile_mask].T
                y_pos = y_pos[torch.arange(y_pos.shape[0]), v_pos]
                bi_out = torch.where(b_out == 0, f_out, b_out).permute(1, 0, 2)
                bi_out[torch.arange(bi_out.shape[0]), v_pos, y_pos] = 1
                bi_out = bi_out.permute(1, 0, 2)
                bi_out = bi_out[1:].reshape(-1, bi_out.shape[-1])

            out_loss = torch.tensor(0).float().to(self.device)
            if early_stage:
                y = y[1:].view(-1)
                profile_exists_loss = torch.tensor(0).float().to(self.device)
                profile_detector_loss = torch.tensor(0).float().to(self.device)
                out_loss = self.out_loss_fn(bi_out, y)
                print('early_stage out_loss:', out_loss.item())
            else:
                # profile_exists_loss = self.profile_exists_loss_fn(profile_exists, profile_y)
                profile_exists_loss = F.binary_cross_entropy(profile_exists, profile_y)
                profile_detector_loss = torch.tensor(0).float().to(self.device)
                if exists_no_profile:
                    out_loss += self.out_loss_fn(
                            out[1:].view(-1, out.shape[-1]), 
                            y[:, no_profile_mask][1:].view(-1))
                    print('exists_no_profile out_loss:', out_loss.item())
                if exists_has_profile:
                    _profile_j = torch.zeros_like(beta).to(self.device)
                    _profile_j[torch.arange(beta.shape[0]), profile_j[has_profile_mask].long()] = 1
                    profile_detector_loss = self.profile_detector_loss_fn(beta, _profile_j)
                    out_loss += self.out_loss_fn(
                            bi_out, 
                            y[:, has_profile_mask][1:].view(-1))
                    print('exists_has_profile out_loss:', out_loss.item())

            print('profile_exists_loss, profile_detector_loss: ', 
                    profile_exists_loss.item(), profile_detector_loss.item())
            loss = out_loss + self.args.alpha * (profile_exists_loss + profile_detector_loss)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()

            iloss = loss.item()
            epoch_loss += iloss
            print(f'Train Loss: {iloss:.3f} | Train PPL: {math.exp(iloss):7.3f}\n')

        return epoch_loss / len(data_iter)

    def eval(self, data_iter=None):
        self.model.eval()
        return 1

        if data_iter is None:
            data_iter = self.test_iter

        epoch_loss = 0
        with torch.no_grad():
            for _, (X, y, key) in enumerate(data_iter):
                f_out, j, b_out = self.model(X, y, teacher_forcing_ratio=0)

                out = out[1:].view(-1, out.shape[-1])
                y = y[1:].view(-1)
                loss = self.loss_fn(out, y)

                epoch_loss += loss.item()

        return epoch_loss / len(data_iter)

    def save_model(self, epoch):
        model_path = os.path.join(self.args.model_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.model.state_dict(), model_path + '/model.pt')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.pretrained_path))
        self.model.eval()


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # all keys and values must in word2vecs,
    # it means they are appeared in corpus when pretrain_emb, or training data
    # TODO: remove profile order dependence
    profiles = OrderedDict(
            姓名='张',
            #姓名='张三丰',
            年龄='三岁',
            性别='男孩',
            #爱好='动漫',
            #特长='钢琴',
            体重='60',
            地址='北京',
            星座='双子座',
    )
    trainer = Trainer(profiles)
    trainer.run()

