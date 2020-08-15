
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
from torch.utils.data import DataLoader
import torch.utils.checkpoint as torch_cp

import torch_optimizer as toptim
import transformers
from transformers import Pipeline, BertTokenizer, AlbertModel, \
        AutoTokenizer, AutoModel, AlbertConfig, AutoConfig


class Args:
    pass

# XXX: change all class init param to context
#      but the code will be hard to reuse and not transparent
class Context:
    def __init__(self):
        self.config = Args()
        self.vocab = None
        self.persona_vocab = None


class Trainer:
    def __init__(self):
        self.context = Context()

        args = self.parse_args()
        self.args = args
        self.best_valid_loss = float('inf')
        self.device = utils.get_device(args.device)
        utils.set_random_seed(self.args.seed, self.device)

        self.logger = utils.create_logger(self.args.log_path, 'trainer')

        self.ensure_deps()

        self.logger.info('Build vocab and embeddings...')
        self.pretrain_feature_model = None
        self.tokenizer = None
        self.persona_vocab = None

        if self.args.persona_emb_dim is not None:
            self.build_persona_vocab()

        if self.args.pretrain_feature:
            self.build_pretrain_feature_model()
        else:
            self.build_vocab_and_embeddings()
        self.logger.info('Build dataloaders...')
        self.build_dataloaders()
        self.logger.info('Build model...')
        self.build_model()
        self.logger.info('Build loss fns...')
        self.build_loss_fns()

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_file', default='configs/large.yaml', type=str, required=False, 
            help='Provide config in config_file or as other commandline args')
        parser.add_argument('--experiment_name', default='', type=str, required=False, help='')
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
        parser.add_argument('--share_encoder_decoder', action='store_true', required=False, help='')
        parser.add_argument('--pretrain_feature', action='store_true', required=False, help='')
        parser.add_argument('--pretrain_feature_model_name', default='', type=str, required=False, help='')
        parser.add_argument('--pretrain_feature_type', default='emb', type=str, required=False, help='')

        parser.add_argument('--emb_freeze', action='store_true', required=False, help='')
        parser.add_argument('--emb_dim', default=200, type=int, required=False, help='')
        parser.add_argument('--persona_emb_dim', default=200, type=int, required=False, help='')
        parser.add_argument('--persona_vocab_size', type=int, required=False, help='Will auto fill')
        parser.add_argument('--dropout', default=0.1, type=float, required=False, help='')
        parser.add_argument('--num_layers', default=6, type=int, required=False, help='')
        parser.add_argument('--num_groups', default=1, type=int, required=False, help='')
        parser.add_argument('--n_head', default=8, type=int, required=False, help='')
        parser.add_argument('--d_model', default=512, type=int, required=False, help='')
        parser.add_argument('--d_ff', default=2048, type=int, required=False, help='')
        parser.add_argument('--attn_alpha', default=1, type=int, required=False, help='')
        parser.add_argument('--adapter_d_ff', default=2048, type=int, required=False, help='')
        parser.add_argument('--factor_ff', action='store_true', required=False, help='')

        parser.add_argument('--lr', default=0.5, type=float, required=False, help='')
        parser.add_argument('--weight_decay', default=0.99, type=float, required=False, help='')
        parser.add_argument('--clip_grad', default=1, type=int, required=False, help='')
        parser.add_argument('--use_scheduler', default=False, type=bool, required=False, help='')
        parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='')
        parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='')
        parser.add_argument('--adapter_finetune', default=False, type=bool, required=False, help='')
        parser.add_argument('--auxiliary_task', default='MLM', type=str, required=False, help='')
        parser.add_argument('--alpha', default=0.5, type=float, required=False, help='LM loss weight')

        parser.add_argument('--model_path', default='models/', type=str, required=False, help='')
        parser.add_argument('--pretrained_fname', type=str, required=False, help='')
        parser.add_argument('--data_path', default='datas/', type=str, required=False, help='')
        parser.add_argument('--cache_path', default='caches/', type=str, required=False, help='')
        parser.add_argument('--log_path', default='logs/', type=str, required=False, help='')
        parser.add_argument('--corpus_fname', default='datas/corpus.txt', type=str, required=False, help='')
        parser.add_argument('--vec_fname', default='models/vec.txt', type=str, required=False, help='')
        parser.add_argument('--vocab_fname', default='models/vocab.txt', type=str, required=False, help='')
        parser.add_argument('--persona_vocab_fname', default='', type=str, required=False, help='')
        parser.add_argument('--lr_finder', action='store_true', required=False, help='')

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

    def build_persona_vocab(self):
        self.persona_vocab = datasets.PersonaVocab(self.args.persona_vocab_fname)
        self.args.persona_vocab_size = len(self.persona_vocab)

    def build_vocab_and_embeddings(self):
        args = self.args

        if args.pretrain_emb and (
                not os.path.exists(args.vec_fname) 
                or not os.path.exists(args.vocab_fname)
        ):
            self.logger.info('Pretraining word2vec...')
            models.build_word2vec(args.corpus_fname, args.vec_fname, args.vocab_fname, args.d_model)

        embeddings, gensim_vocab = None, None
        if args.pretrain_emb:
            self.logger.info('Loading word2vec...')
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

    def build_pretrain_feature_model(self):
        mn = self.args.pretrain_feature_model_name
        if 'albert' in mn:
            pretrain_feature_tokenizer = BertTokenizer.from_pretrained(mn)
            config = AlbertConfig.from_pretrained(mn)
            config.output_hidden_states = True
            self.pretrain_feature_model = AlbertModel.from_pretrained(mn,
                    config=config).to(self.device)
        else:
            pretrain_feature_tokenizer = AutoTokenizer.from_pretrained(mn)
            config = AutoConfig.from_pretrained(mn)
            config.output_hidden_states = True
            self.pretrain_feature_model = AutoModel.from_pretrained(mn,
                    config=config).to(self.device)
        self.pretrain_feature_model.requires_grad_(False)
        # self.pretrain_feature_model.requires_grad_(True)
        # pipeline input is raw data, we have ids, so direct use model
        # self.pretrain_feature_pipeline = Pipeline('feature-extraction', 
        #        model=self.pretrain_feature_model, tokenizer=pretrain_feature_tokenizer)

        # TODO: pre calc feature and save to file, it use less memory for train and faster 
        # XXX: only used this tokenizer vocab, did not used for byte pair split, now just split by space
        utils.add_special_tokens_(self.pretrain_feature_model, pretrain_feature_tokenizer)
        # FIXME: this changed args should saved to checkpoint file
        if self.args.pretrain_feature_type == 'mem_n2n': 
            self.args.emb_dim = self.pretrain_feature_model.config.hidden_size
            self.args.d_model = self.pretrain_feature_model.config.hidden_size
        elif self.args.pretrain_feature_type == 'feature':
            self.args.emb_dim = self.pretrain_feature_model.config.hidden_size
        else:
            if self.pretrain_feature_model.base_model_prefix != 'bert':
                self.args.emb_dim = self.pretrain_feature_model.config.embedding_size
            else:
                self.args.emb_dim = self.pretrain_feature_model.config.hidden_size

        # XXX: for 'xlnet'
        # self.args.d_model = self.pretrain_feature_model.config.hidden_size

        if 'weight' in self.args.pretrain_feature_type:
            # few effects
            self.args.d_model = self.pretrain_feature_model.config.hidden_size
            self.args.n_head = self.pretrain_feature_model.config.num_attention_heads
            self.args.d_ff = self.pretrain_feature_model.config.intermediate_size
            self.args.factor_ff = False

        self.vocab = datasets.ChatVocab(pretrain_feature_tokenizer)
        self.input_dim = len(self.vocab)
        self.pad_idx = self.vocab.stoi(utils.PAD)
        self.embeddings = None
        # too slow
        # self.tokenizer = pretrain_feature_tokenizer.tokenize
        self.tokenizer = None
                                                                                         
    def build_dataloaders(self):
        args = self.args
        gb = lambda batch: datasets.generate_batch(batch, self.vocab, self.persona_vocab)
        gb_lm = lambda batch: datasets.generate_lm_batch(batch, self.vocab)

        if args.n_epochs_early_stage > 0:
            dp = datasets.LMDataProcesser(limit_length=args.limit_example_length, 
                    max_seq_length=args.max_seq_length,
                    tokenizer=self.tokenizer)
            ds = utils.PersonaDataset(
                    self.vocab, args.max_seq_length, args.limit_example_length,
                    data_path=args.data_path, cache_path=args.cache_path, 
                    data_processer=dp, mode='train_lm')
            self.logger.info('---------------------------------')
            self.logger.info('datasets len: %s' % len(ds))
            self.train_iter = DataLoader(ds, batch_size=args.batch_size, 
                    collate_fn=gb_lm, shuffle=True) 
        else:
            dp = datasets.ChatDataProcesser(limit_length=args.limit_example_length, 
                    max_seq_length=args.max_seq_length, max_context_size=args.max_context_size,
                    vocab=self.vocab, persona_vocab=self.persona_vocab,
                    tokenizer=self.tokenizer)
            ds = utils.PersonaDataset(
                    self.vocab, args.max_seq_length, args.limit_example_length, 
                    data_path=args.data_path, cache_path=args.cache_path, 
                    data_processer=dp, mode='train_char')
            self.logger.info('---------------------------------')
            self.logger.info('datasets len: %s' % len(ds))
            # when Dataset is stream, try utils.DataLoaderX (prefetch_generator), https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
            self.train_iter = DataLoader(ds, batch_size=args.batch_size, 
                    collate_fn=gb, shuffle=args.shuffle_data) 

            dp = datasets.ChatDataProcesser(limit_length=args.limit_example_length, 
                        max_seq_length=args.max_seq_length, max_context_size=args.max_context_size,
                        vocab=self.vocab, persona_vocab=self.persona_vocab,
                        tokenizer=self.tokenizer)
            ds = utils.PersonaDataset(
                    self.vocab, args.max_seq_length, args.limit_example_length, 
                    data_path=args.data_path, cache_path=args.cache_path, 
                    data_processer=dp, mode='valid_char')
            self.valid_iter = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=gb, shuffle=False) 

            dp = datasets.ChatDataProcesser(limit_length=args.limit_example_length, 
                        max_seq_length=args.max_seq_length, max_context_size=args.max_context_size,
                        vocab=self.vocab, persona_vocab=self.persona_vocab,
                        tokenizer=self.tokenizer)
            ds = utils.PersonaDataset(
                    self.vocab, args.max_seq_length, args.limit_example_length, 
                    data_path=args.data_path, cache_path=args.cache_path, 
                    data_processer=dp, mode='test_char')
            self.test_iter = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=gb, shuffle=False)

    def build_model(self):
        args = self.args
        output_dim = self.input_dim
        input_dim = self.input_dim

        self.best_model = None
        # TODO: change all modules param to single config, 
        #       change input_dim and output_dim to args.vocab_size
        self.model = models.AR.build(args, input_dim, 
                output_dim, self.vocab, self.embeddings,
                self.pretrain_feature_model).to(self.device)

        self.optimizer = transformers.AdamW(self.model.parameters(), lr=args.lr, correct_bias=True,
        #self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr,
        #self.optimizer = toptim.Lamb(self.model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)
        self.logger.info(self.model)
        self.logger.info(f'The model has {utils.count_parameters(self.model):,} trainable parameters') 

        if args.use_scheduler:
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
            #self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 2)
            if args.warmup_steps == 0:
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                        mode='min', factor=0.5, min_lr=1.5e-4, patience=60, verbose=True)
            else:
                # XXX: scheduler will run once at start, even if has no scheduler.step()
                total_steps = int(len(self.train_iter.dataset) * args.n_epochs 
                        / args.batch_size / args.gradient_accumulation)
                self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, 
                        num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
 
        if args.pretrained_fname is None:
            pass
            # pytorch module will auto init_weights with uniform
            # self.model.apply(models.init_weights)
        else:
            self.logger.info()
            self.logger.info(f'Load pretrained model {args.pretrained_fname}...')
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

            self.logger.info('-' * 89)
            self.logger.info('Experiment %s: ' % self.args.experiment_name)
            self.logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            self.logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    def run(self):
        if self.args.n_epochs_early_stage > 0:
            self.logger.info('Run early stage...')
            trainer.run_early_stage()
            # after fin, rerun with pretrained model 
            return

        self.logger.info('Run main stage...')

        best_val_loss = float("inf")

        for epoch in range(self.args.n_epochs):
            start_time = time.time()

            train_loss = self.train(epoch)
            valid_loss = self.eval(self.valid_iter)
 
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                self.best_model = self.model
                self.save_model(epoch)

            # scheduler.step()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            self.logger.info('-' * 89)
            self.logger.info('Experiment %s: ' % self.args.experiment_name)
            self.logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            self.logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            self.logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        test_loss = self.eval(self.test_iter)
        self.logger.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

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
            # utils.self.logger.info_backward_graph(loss)
            loss.backward()

            if self.args.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()

            iloss = loss.item()
            epoch_loss += iloss

            end_time = time.time()
            secs = end_time - start_time
            self.logger.info(f'Step {batch_idx+1}/{epoch+1:02} | Train Loss: {iloss:.3f} | Train PPL: {math.exp(iloss):7.3f} | Time: {secs:.3f}s\n')

        return epoch_loss / len(self.train_iter)
 

    def train(self, epoch, data_iter=None):
        self.model.train()

        if data_iter is None:
            data_iter = self.train_iter

        epoch_loss = 0
        for batch_idx, feature in enumerate(data_iter):
            start_time = time.time()

            utils.feature_to_device(feature, self.device)

            # out, out_lm = torch_cp.checkpoint(self.model, feature)
            out, out_lm = self.model(feature)
            loss, loss_lm = models.AR.loss(self.args.auxiliary_task, 
                    self.out_loss_fn, out, out_lm, feature.resp, feature.lm.y)
            if self.args.auxiliary_task is not None:
                loss = loss + self.args.alpha * loss_lm
            if self.args.gradient_accumulation > 1:
                loss = loss / self.args.gradient_accumulation
                # accuracy = accuracy / self.args.gradient_accumulation
            # utils.self.logger.info_backward_graph(loss)
            loss.backward()

            iloss = loss.item()
            epoch_loss += iloss

            if self.args.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

            if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.args.use_scheduler:
                    self.scheduler.step(iloss)

                end_time = time.time()
                secs = end_time - start_time
                self.logger.info(f'Step {batch_idx+1}/{epoch+1:02} | Train Loss: {iloss:.3f} | Train PPL: {math.exp(iloss):7.3f} | Time: {secs:.3f}s\n')

        return epoch_loss / len(data_iter)

    def eval(self, data_iter):
        self.model.eval()

        return 0

        epoch_loss = 0
        with torch.no_grad():
            for _, feature in enumerate(data_iter):

                utils.feature_to_device(feature, self.device)

                out, out_lm = self.model(feature)
                loss, loss_lm = models.AR.loss(self.args.auxiliary_task,
                        self.out_loss_fn, out, out_lm, feature.resp, feature.lm.y)
                if self.args.auxiliary_task is not None:
                    loss = loss + self.args.alpha * loss_lm

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
        shutil.copyfile(self.args.config_file, model_path + '/config.yml')
        shutil.copyfile(self.args.vocab_fname, model_path + '/vocab')

    def load_model(self):
        # tmp for load pretrain LM model before factor ff
        state_dict = torch.load(self.args.pretrained_fname)
       #state_dict = {k: v 
       #        for k, v in state_dict.items() 
       #        if 'linear1' not in k and 'linear2' not in k}

        self.model.load_state_dict(state_dict, strict=False)


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def lr_finder(trainer):
    import matplotlib
    import matplotlib.pyplot as plt
    import torch_lr_finder
    matplotlib.use('WebAgg') 

    class TrainIter(torch_lr_finder.TrainDataLoaderIter): 
        def inputs_labels_from_batch(self, batch_data): 
            utils.feature_to_device(batch_data, 'cuda') 
            return (batch_data, (batch_data.resp, batch_data.lm)) 

    def loss_fn(outputs, target):
        alpha = trainer.args.alpha
        out, out_lm = outputs
        loss, loss_lm = models.AR.loss(trainer.args.auxiliary_task, 
                trainer.out_loss_fn, out, out_lm, target[0], target[1].y)
        loss = loss + alpha * loss_lm
        return loss

    lr = torch_lr_finder.LRFinder(trainer.model, trainer.optimizer, loss_fn, device='cuda')
    lr.range_test(TrainIter(trainer.train_iter), end_lr=100, num_iter=100)
    lr.plot()
    lr.reset()
    # plt.savefig("mygraph.png")
    # plt.show()
    print(lr.history)


if __name__ == '__main__':
    trainer = Trainer()
    if trainer.args.lr_finder:
        lr_finder(trainer)
    else:
        trainer.run()

