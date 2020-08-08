
import os
import copy
import time
import math
import argparse
import yaml
import sys
# for import parent utils
sys.path.append('../')
import utils

import models
import metrics
import datasets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Pipeline, BertTokenizer, AlbertModel, \
        AutoTokenizer, AutoModel, AlbertConfig, AutoConfig


class Evaler:
    def __init__(self):
        args = self.parse_args()
        self.args = args
        self.device = utils.get_device(args.device)
        utils.set_random_seed(self.args.seed, self.device)

        self.logger = utils.create_logger(self.args.log_path, 'evaler')

        self.model_config = self.load_model_config()
        self.args.max_seq_length = self.model_config.max_seq_length

        self.logger.info('Load vocab...')
        if self.model_config.pretrain_feature:
            self.build_pretrain_feature_model()
        else:
            self.pretrain_feature_model = None
            self.build_vocab()
        self.logger.info('Build dataloaders...')
        self.build_dataloaders()
        self.logger.info('Build model...')
        self.build_model()
        self.logger.info('Build loss fns...')
        self.build_loss_fns()

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_file', default='configs/evaler.yaml', type=str, required=False, 
            help='Provide config in config_file or as other commandline args')
        parser.add_argument('--device', default='cuda', type=str, required=False, help='use cpu for easy debug')
        parser.add_argument('--seed', default=42, type=int, required=False, help='')
        parser.add_argument('--batch_size', default=128, type=int, required=False, help='')
        parser.add_argument('--limit_example_length', default=256, type=int, required=False, help='')

        parser.add_argument('--min_seq_length', default=10, type=int, required=False, help='')
        parser.add_argument('--temperature', default=0.7, type=float, required=False, help='Sampling softmax temperature')
        parser.add_argument('--top_k', default=0, type=int, required=False, help='Filter top-k tokens before sampling (<=0: no filtering)')
        parser.add_argument('--top_p', default=0.9, type=float, required=False, help='Nucleus filtering (top-p) before sampling (<=0.0: no filtering)')
        parser.add_argument('--no_sample', action='store_true', required=False, help='')

        parser.add_argument('--pretrained_fname', default='models/model__epoch1/', type=str, required=False, help='')
        parser.add_argument('--data_path', default='datas/', type=str, required=False, help='')
        parser.add_argument('--cache_path', default='caches/', type=str, required=False, help='')
        parser.add_argument('--log_path', default='logs/', type=str, required=False, help='')

        # TODO: let commandline temp args override args in config_file
        args = parser.parse_args()
        if args.config_file != '':
            parser.set_defaults(**yaml.load(open(args.config_file)))
            args = parser.parse_args()

        return args

    def build_vocab(self):
        args = self.args

        # TODO: seperate embeddings load from vocab
        _, gensim_vocab = utils.load_embeddings_and_vocab(self.model_config.vec_fname, 
                self.get_model_deps_file('vocab'))
        self.vocab = utils.Vocab(gensim_vocab, args.data_path)
        self.input_dim = len(self.vocab)
        self.pad_idx = self.vocab.stoi(utils.PAD)

    def build_pretrain_feature_model(self):
        mn = self.model_config.pretrain_feature_model_name
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
        # pipeline input is raw data, we have ids, so direct use model
        # self.pretrain_feature_pipeline = Pipeline('feature-extraction', 
        #        model=self.pretrain_feature_model, tokenizer=pretrain_feature_tokenizer)

        # TODO: pre calc feature and save to file, it use less memory for train and faster 
        # XXX: only used this tokenizer vocab, did not used for byte pair split, now just split by space
        utils.add_special_tokens_(self.pretrain_feature_model, pretrain_feature_tokenizer)
        # FIXME: this changed args should saved to checkpoint file
        # for use feature
        # self.args.emb_dim = self.pretrain_feature_model.config.hidden_size
        # self.model_config.emb_dim = self.pretrain_feature_model.config.hidden_size
        # for use emb
        self.args.emb_dim = self.pretrain_feature_model.config.embedding_size
        self.model_config.emb_dim = self.pretrain_feature_model.config.embedding_size

        # for use layer weight
        self.model_config.d_model = self.pretrain_feature_model.config.hidden_size

        self.vocab = datasets.ChatVocab(pretrain_feature_tokenizer)
        self.input_dim = len(self.vocab)
        self.pad_idx = self.vocab.stoi(utils.PAD)

        # pretrain_feature_model emb and weight no need anymore, use trained model
        self.pretrain_feature_model = None
    
    def build_dataloaders(self):
        args = self.args
        model_config = self.model_config
        gb = lambda batch: datasets.generate_batch(batch, self.vocab)

        dp = datasets.ChatDataProcesser(limit_length=args.limit_example_length, 
                    max_seq_length=model_config.max_seq_length, 
                    max_context_size=model_config.max_context_size)
        ds = utils.PersonaDataset(
                self.vocab, model_config.max_seq_length, args.limit_example_length, 
                data_path=args.data_path, cache_path=args.cache_path, 
                data_processer=dp, mode='test_char')
        self.test_iter = DataLoader(ds, batch_size=args.batch_size,
                collate_fn=gb, shuffle=False)
    
    def build_model(self):
        args = self.args
        output_dim = self.input_dim
        input_dim = self.input_dim

        self.model = models.AR.build(self.model_config, input_dim, 
                output_dim, self.vocab, None,
                self.pretrain_feature_model).to(self.device)

        print(f'Load pretrained model {args.pretrained_fname}...')
        self.load_model()

    def build_loss_fns(self):
        self.out_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def run(self):
        self.model.eval()

        total_bleu = 0
        total_f1 = 0
        total_dist1 = 0
        total_dist2 = 0
        total_loss = 0

        print('Run eval...')
        with torch.no_grad():
            for batch_idx, feature in enumerate(self.test_iter):
                utils.feature_to_device(feature, self.device)

                out, out_lm = self.model(feature)
                print(out[0, 0], out_lm[0, 0])
                loss, loss_lm = models.AR.loss(self.out_loss_fn, out, out_lm, feature.resp, feature.lm.y)
                print(loss, loss_lm)
                loss = loss + self.model_config.alpha * loss_lm
                total_loss += loss.item()

                # target include w1, w2...[EOS], len: max_seq_length + 1
                target = copy.deepcopy(feature.resp[1:])
                # feature will be changed
                pred, pred_padded = utils.sample_sequence(feature, self.vocab, self.model, self.args)

                pred_tokens = [[self.vocab.itos(k) for k in ks]
                               for ks in pred]
                target_tokens = [[[self.vocab.itos(k) for k in ks]]
                                 for ks in target.T.tolist()]
                print('----------------------------------')
                print('Pred: ', ''.join([self.vocab.itos(k)
                                 for k in pred_padded.T.tolist()[0]]))
                print('Target: ', ''.join(target_tokens[0][0]))
                print('Pred: ', ''.join([self.vocab.itos(k)
                                 for k in pred_padded.T.tolist()[-1]]))
                print('Target: ', ''.join(target_tokens[-1][0]))
                print('----------------------------------')
                bleu = metrics.bleu_score(pred_tokens, target_tokens)
                f1 = metrics.f1_score(pred_padded.T.to('cpu'), target.T.to('cpu'))
                # dist1 = metrics.distinct_score([v[:-1] for v in pred])
                dist1 = metrics.distinct_score(pred_tokens)
                dist2 = metrics.distinct_score(pred_tokens, 2)

                total_bleu += bleu
                total_f1 += f1
                total_dist1 += dist1
                total_dist2 += dist2

        l = len(self.test_iter)
        bleu = total_bleu/l
        f1 = total_f1/l
        dist1 = total_dist1/l
        dist2 = total_dist2/l
        # https://stackoverflow.com/questions/59209086/calculate-perplexity-in-pytorch
        # see per-word perplexity:
        # https://github.com/huggingface/transfer-learning-conv-ai/blob/master/convai_evaluation.py#L161
        # https://github.com/facebookresearch/ParlAI/blob/56d46551190a7ffaedccd13534412d43bc7076e5/parlai/scripts/eval_ppl.py
        ppl = math.exp(total_loss/l)

        print(f'\tBleu: {bleu:.8f} | F1: {f1:.8f} | '
              f'Dist1: {dist1:.3f} | Dist2: {dist2:.3f} | PPL: {ppl:7.3f}')

    def load_model_config(self):
        class Args:
            pass

        args = Args()
        d = yaml.load(open(self.get_model_deps_file('config.yml')))
        for k, v in d.items():
            setattr(args, k, v)

        return args

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.pretrained_fname))

    def get_model_deps_file(self, fname):
        return os.path.dirname(self.args.pretrained_fname) + '/' + fname


if __name__ == '__main__':
    evaler = Evaler()
    evaler.run()
