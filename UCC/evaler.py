
import os
import copy
import sys
# for import parent utils
sys.path.append('../')
import utils

import models
import metrics


class Evaler:
    def __init__(self):
        args = self.parse_args()
        self.args = args
        self.device = utils.get_device(args.device)
        utils.set_random_seed(self.seed, self.device)

        self.model_config = self.load_model_config()

        print('Load vocab...')
        self.build_vocab()
        print('Build dataloaders...')
        self.build_dataloaders()
        print('Build model...')
        self.build_model()

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_file', default='configs/evaler.yaml', type=str, required=False, 
            help='Provide config in config_file or as other commandline args')
        parser.add_argument('--device', default='cuda', type=str, required=False, help='use cpu for easy debug')
        parser.add_argument('--seed', default=42, type=int, required=False, help='')
        parser.add_argument('--batch_size', default=128, type=int, required=False, help='')
        parser.add_argument('--limit_example_length', default=256, type=int, required=False, help='')
        parser.add_argument('--max_seq_length', default=300, type=int, required=False, help='')
        parser.add_argument('--max_context_size', default=10, type=int, required=False, help='')

        parser.add_argument('--pretrained_fname', default='models/model__epoch1/', type=str, required=False, help='')
        parser.add_argument('--data_path', default='datas/', type=str, required=False, help='')
        parser.add_argument('--cache_path', default='caches/', type=str, required=False, help='')

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
    
    def build_dataloaders(self):
        args = self.args
        gb = lambda batch: datasets.generate_batch(batch, self.pad_idx)

        dp = datasets.ChatDataProcesser(limit_length=args.limit_example_length, 
                    max_seq_length=args.max_seq_length, max_context_size=args.max_context_size)
        ds = utils.PersonaDataset(
                self.vocab, args.max_seq_length, args.limit_example_length, 
                data_path=args.data_path, cache_path=args.cache_path, 
                data_processer=dp, mode='test_char')
        self.test_iter = DataLoader(ds, batch_size=args.batch_size,
                collate_fn=gb, shuffle=True)
 
    
    def build_model(self):
        args = self.args
        output_dim = self.input_dim
        input_dim = self.input_dim

        self.model = models.AR.build(self.model_config, input_dim, 
                output_dim, self.vocab)

        print(f'Load pretrained model {args.pretrained_fname}...')
        self.load_model()

    def run(self):
        self.model.eval()

        total_bleu = 0
        total_f1 = 0
        total_dist1 = 0
        total_loss = 0
        with torch.no_grad():
            for batch_idx, feature in enumerate(self.test_iter):
                utils.feature_to_device(feature, self.device)

                out, out_lm = self.model(feature)
                loss, loss_lm = models.AR.loss(self.out_loss_fn, out, out_lm, feature.resp, feature.lm.y)
                loss = loss + self.model_config.alpha * loss_lm
                total_loss += loss.item()

                target = copy.deepcopy(feature.resp[1:].view(-1))
                # feature may be changed
                pred = utils.sample_sequence(feature, self.vocab, self.model, self.args)

                bleu = metrics.bleu_score(pred, target)
                f1 = metrics.f1_score(pred, target)
                dist1 = metrics.distinct_score(pred)

                total_bleu += bleu
                total_f1 += f1
                total_dist1 += dist1

        l = len(self.test_iter)
        bleu = total_bleu/l
        f1 = total_f1/l
        dist1 = total_dist1/l
        # https://stackoverflow.com/questions/59209086/calculate-perplexity-in-pytorch
        # see per-word perplexity:
        # https://github.com/huggingface/transfer-learning-conv-ai/blob/master/convai_evaluation.py#L161
        # https://github.com/facebookresearch/ParlAI/blob/56d46551190a7ffaedccd13534412d43bc7076e5/parlai/scripts/eval_ppl.py
        ppl = math.exp(total_loss/l)

        print(f'\tBleu: {bleu:.3f} | F1: {f1:.3f} | Dist1: {dist1:.3f} | PPL: {ppl:7.3f}')

    def load_model_config(self):
        return yaml.load(open(self.get_model_deps_file('config.yml')))

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.pretrained_fname))

    def get_model_deps_file(self, fname):
        return os.path.dirname(self.args.pretrained_fname) + '/' + fname


if __name__ == '__main__':
    evaler = Evaler()
    evaler.run()
