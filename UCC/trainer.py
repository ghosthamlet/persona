
import os
import shutil
import random
import math
import time
import argparse
import yaml
from collections import OrderedDict

import sys
# for import parent utils
sys.path.append('../')
import utils
import modules
import datasets
import models

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import torch_optimizer as toptim
import transformers


class Trainer:
    def __init__(self):
        args = self.parse_args()
        self.args = args
        self.best_valid_loss = float('inf')
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        self.set_random_seed()

        self.ensure_deps()

        print('Build vocab and embeddings...')
        self.build_vocab_and_embeddings()
        print('Build dataloaders...')
        self.build_dataloaders()
        print('Build model...')
        self.build_model()
        print('Build loss fns...')
        self.build_loss_fns()

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
        parser.add_argument('--device', default='cuda', type=str, required=False, help='use cpu for easy debug')
        parser.add_argument('--seed', default=42, type=int, required=False, help='')
        parser.add_argument('--n_epochs', default=10, type=int, required=False, help='')
        parser.add_argument('--n_epochs_early_stage', default=0, type=int, required=False, help='')
        parser.add_argument('--batch_size', default=128, type=int, required=False, help='')
        parser.add_argument('--limit_example_length', default=256, type=int, required=False, help='')
        parser.add_argument('--max_seq_length', default=300, type=int, required=False, help='')
        parser.add_argument('--max_context_size', default=10, type=int, required=False, help='')
        parser.add_argument('--shuffle_data', action='store_true', required=False, help='')
        parser.add_argument('--max_vocab_size', default=40000, type=int, required=False, help='')
        parser.add_argument('--pretrain_emb', action='store_true', required=False, help='')

        parser.add_argument('--emb_freeze', action='store_true', required=False, help='')
        parser.add_argument('--dropout', default=0.1, type=float, required=False, help='')
        parser.add_argument('--num_layers', default=6, type=int, required=False, help='')
        parser.add_argument('--n_head', default=8, type=int, required=False, help='')
        parser.add_argument('--d_model', default=512, type=int, required=False, help='')
        parser.add_argument('--d_ff', default=2048, type=int, required=False, help='')
        parser.add_argument('--attn_alpha', default=1, type=int, required=False, help='')
        parser.add_argument('--adapter_d_ff', default=2048, type=int, required=False, help='')

        parser.add_argument('--lr', default=0.5, type=float, required=False, help='')
        parser.add_argument('--weight_decay', default=0.99, type=float, required=False, help='')
        parser.add_argument('--clip_grad', default=1, type=int, required=False, help='')
        parser.add_argument('--use_scheduler', default=False, type=bool, required=False, help='')
        parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='')
        parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='')
        parser.add_argument('--adapter_finetune', default=False, type=bool, required=False, help='')

        parser.add_argument('--model_path', default='models/', type=str, required=False, help='')
        parser.add_argument('--pretrained_path', type=str, required=False, help='')
        parser.add_argument('--data_path', default='datas/', type=str, required=False, help='')
        parser.add_argument('--cache_path', default='caches/', type=str, required=False, help='')
        parser.add_argument('--corpus_fname', default='datas/corpus.txt', type=str, required=False, help='')
        parser.add_argument('--vec_fname', default='models/vec.txt', type=str, required=False, help='')
        parser.add_argument('--vocab_fname', default='models/vocab.txt', type=str, required=False, help='')

        # TODO: let commandline temp args override args in config_file
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

    def build_vocab_and_embeddings(self):
        args = self.args

        if args.pretrain_emb and (
                not os.path.exists(args.vec_fname) 
                or not os.path.exists(args.vocab_fname)
        ):
            print('Pretraining word2vec...')
            models.build_word2vec(args.corpus_fname, args.vec_fname, args.vocab_fname, args.d_model)

        embeddings, gensim_vocab = None, None
        if args.pretrain_emb:
            print('Loading word2vec...')
            embeddings, gensim_vocab = utils.load_embeddings_and_vocab(args.vec_fname, args.vocab_fname)
            embeddings = embeddings.to(self.device)
        self.vocab = utils.Vocab(gensim_vocab, args.data_path)
        self.input_dim = len(self.vocab)
        if args.pretrain_emb:
            elen = embeddings.shape[0]
            if self.input_dim > elen:
                args.emb_freeze = False
                append = torch.nn.init.kaiming_uniform_(
                        torch.zeros(self.input_dim - elen, embeddings.shape[1])).to(self.device)
                embeddings = torch.cat([embeddings, append], dim=0)

        self.pad_idx = self.vocab.stoi(utils.PAD)
        self.embeddings = embeddings
                                                                                         
    def build_dataloaders(self):
        args = self.args
        gb = lambda batch: datasets.generate_batch(batch, self.pad_idx)
        gb_lm = lambda batch: datasets.generate_lm_batch(batch, self.pad_idx)

        if args.n_epochs_early_stage > 0:
            dp = datasets.LMDataProcesser(limit_length=args.limit_example_length, 
                    max_seq_length=args.max_seq_length)
            ds = utils.PersonaDataset(
                    self.vocab, args.max_seq_length, args.limit_example_length,
                    data_path=args.data_path, cache_path=args.cache_path, 
                    data_processer=dp, mode='train_lm')
            print('---------------------------------')
            print('datasets len:', len(ds))
            self.train_iter = DataLoader(ds, batch_size=args.batch_size, 
                    collate_fn=gb_lm, shuffle=True) 
        else:
            dp = datasets.ChatDataProcesser(limit_length=args.limit_example_length, 
                    max_seq_length=args.max_seq_length, max_context_size=args.max_context_size)
            ds = utils.PersonaDataset(
                    self.vocab, args.max_seq_length, args.limit_example_length, 
                    data_path=args.data_path, cache_path=args.cache_path, 
                    data_processer=dp, mode='train_char')
            print('---------------------------------')
            print('datasets len:', len(ds))
            # when Dataset is stream, try utils.DataLoaderX (prefetch_generator), https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
            self.train_iter = DataLoader(ds, batch_size=args.batch_size, 
                    collate_fn=gb, shuffle=args.shuffle_data) 

        self.valid_iter = None
        self.test_iter = None
       #ds = utils.PersonaDataset(
       #        self.vocab, args.max_seq_length, args.limit_example_length, 
       #        data_path=args.data_path, cache_path=args.cache_path, 
       #        limit_length=args.limit_example_length, mode='valid')
       #self.valid_iter = DataLoader(ds, batch_size=args.batch_size,
       #        collate_fn=gb, shuffle=args.shuffle_data) 

       #ds = utils.PersonaDataset(
       #        self.vocab, args.max_seq_length, args.limit_example_length, 
       #        data_path=args.data_path, cache_path=args.cache_path, 
       #        limit_length=args.limit_example_length, mode='test')
       #self.test_iter = DataLoader(ds, batch_size=args.batch_size,
       #        collate_fn=gb, shuffle=args.shuffle_data)

    def build_model(self):
        args = self.args
        output_dim = self.input_dim
        input_dim = self.input_dim
        pad_idx = self.pad_idx
        embeddings = self.embeddings
        sep_idx = self.vocab.stoi(utils.SEP)
        spe1_idx = self.vocab.stoi(utils.SPE1)
        spe2_idx = self.vocab.stoi(utils.SPE2)

        context_emb = modules.ContextEmb(sep_idx, spe1_idx, spe2_idx,
                input_dim, args.emb_dim, args.emb_freeze, 
                args.d_model, pad_idx, args.dropout, embeddings)
        persona_emb = modules.PersonaEmb(input_dim, args.emb_dim, args.emb_freeze,
                args.d_model, pad_idx, args.dropout, embeddings)
        output_emb = modules.OutputEmb(input_dim, args.emb_dim, args.emb_freeze,
                args.d_model, pad_idx, args.dropout, embeddings)

        post_encoder = modules.TransformerEncoder(input_dim, args.d_model, args.d_ff, 
                args.n_head, args.num_layers, args.dropout,
                'relu', args.adapter_finetune, args.adapter_d_ff)

        resp_decoder_layer = modules.TransformerDecoderLayer(args.d_model, args.n_head, 
                args.attn_alpha, args.d_ff, args.dropout, 
                'relu', args.adapter_finetune, args.adapter_d_ff)
        resp_decoder = modules.TransformerDecoder(resp_decoder_layer, args.num_layers)
        generater = modules.Generater(args.emb_dim, args.d_model, output_dim)

        self.best_model = None
        if args.n_epochs_early_stage > 0:
            self.model = models.LM(
                    context_emb, persona_emb, output_emb,
                    post_encoder, resp_decoder, generater,
                    args.adapter_finetune,
                    ).to(self.device)
            self.optimizer = transformers.AdamW(self.model.parameters(), lr=args.lr, correct_bias=True,
            #self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr,
            #self.optimizer = toptim.Lamb(self.model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)
            print(self.model)
            print(f'The model has {utils.count_parameters(self.model):,} trainable parameters')
        else:
            self.model = models.AR(
                    context_emb, persona_emb, output_emb,
                    post_encoder, resp_decoder, generater,
                    args.adapter_finetune,
                    ).to(self.device)
            self.optimizer = transformers.AdamW(self.model.parameters(), lr=args.lr, correct_bias=True,
            #self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr,
            #self.optimizer = toptim.Lamb(self.model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)
            print(self.model)
            print(f'The model has {utils.count_parameters(self.model):,} trainable parameters')
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 2)
        # XXX: scheduler will run once at start, even if has no scheduler.step()
        if args.use_scheduler:
            total_steps = int(len(self.train_iter.dataset) * args.n_epochs 
                    / args.batch_size / args.gradient_accumulation)
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, 
                    num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


        if args.pretrained_path is None:
            pass
            # pytorch module will auto init_weights with uniform
            # self.model.apply(models.init_weights)
        else:
            print()
            print(f'Load pretrained model {args.pretrained_path}...')
            self.load_model()

    def build_loss_fns(self):
        self.out_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def run_early_stage(self):
        for epoch in range(self.args.n_epochs_early_stage):
            start_time = time.time()

            train_loss = self.train_lm(epoch)
            self.save_model(epoch, 'lm')

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    def run(self):
        if self.args.n_epochs_early_stage > 0:
            print('Run early stage...')
            trainer.run_early_stage()
            # after fin, rerun with pretrained model 
            return

        print('Run main stage...')

        best_val_loss = float("inf")

        for epoch in range(self.args.n_epochs):
            start_time = time.time()

            train_loss = self.train(epoch)
            valid_loss = self.eval()
 
            # if valid_loss < best_val_loss:
                # best_val_loss = valid_loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                self.best_model = self.model
                self.save_model(epoch)

            # scheduler.step()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print('-' * 89)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        test_loss = self.eval(self.test_iter)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    def train_lm(self, epoch):
        self.model.train()

        epoch_loss = 0
        for batch_idx, feature in enumerate(self.train_iter):
            start_time = time.time()
            self.optimizer.zero_grad()

            utils.feature_to_device(feature, self.device)

            out = self.model(feature)
            loss = self.out_loss_fn(out.view(-1, out.shape[-1]), 
                    feature.y.view(-1))
            # utils.print_backward_graph(loss)
            loss.backward()

            if self.args.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()

            iloss = loss.item()
            epoch_loss += iloss

            end_time = time.time()
            secs = end_time - start_time
            print(f'Step {batch_idx+1}/{epoch+1:02} | Train Loss: {iloss:.3f} | Train PPL: {math.exp(iloss):7.3f} | Time: {secs:.3f}s\n')

        return epoch_loss / len(self.train_iter)
 

    def train(self, epoch, data_iter=None):
        self.model.train()

        if data_iter is None:
            data_iter = self.train_iter

        epoch_loss = 0
        for batch_idx, feature in enumerate(data_iter):
            start_time = time.time()

            utils.feature_to_device(feature, self.device)

            alpha = 0.5
            out, out_lm = self.model(feature)
            loss = self.out_loss_fn(out[:-1].view(-1, out.shape[-1]), 
                    feature.resp[1:].view(-1))
            loss_lm = alpha * self.out_loss_fn(out_lm.view(-1, out.shape[-1]), 
                                feature.lm.y.view(-1))
            loss += loss_lm
            if self.args.gradient_accumulation > 1:
                loss = loss / self.args.gradient_accumulation
            #    # accuracy = accuracy / self.args.gradient_accumulation
            # utils.print_backward_graph(loss)
            loss.backward()
            iloss = loss.item()
            epoch_loss += iloss

            if self.args.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

            if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.args.use_scheduler:
                    self.scheduler.step()

                end_time = time.time()
                secs = end_time - start_time
                print(f'Step {batch_idx+1}/{epoch+1:02} | Train Loss: {iloss:.3f} | Train PPL: {math.exp(iloss):7.3f} | Time: {secs:.3f}s\n')

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

    # resuming vs Warmstarting(transfer learning)?
    # it just have difference of optimizer state_dict
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    def save_model(self, epoch, stage=''):
        model_path = os.path.join(self.args.model_path, 
                'model_{}_epoch{}'.format(stage, epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.model.state_dict(), model_path + '/model.pt')
        shutil.copyfile(self.args.config_file, 
                model_path + '/' + os.path.basename(self.args.config_file))

    def load_model(self):
        # tmp for load pretrain LM model before factor ff
        state_dict = torch.load(self.args.pretrained_path)
       #state_dict = {k: v 
       #        for k, v in state_dict.items() 
       #        if 'linear1' not in k and 'linear2' not in k}

        self.model.load_state_dict(state_dict, strict=False)


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()

