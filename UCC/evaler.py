
import sys
# for import parent utils
sys.path.append('../')
import utils

import metrics


class Evaler:
    def __init__(self):
        args = self.parse_args()
        self.args = args
        self.device = utils.get_device(args.device)
        utils.set_random_seed(self.seed, self.device)

        print('Load vocab...')
        self.build_vocab()
        print('Build dataloaders...')
        self.build_dataloaders()
        print('Build model...')
        self.build_model()
        print('Build loss fns...')
        self.build_metric_fns()

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
        parser.add_argument('--pretrain_emb', action='store_true', required=False, help='')

        parser.add_argument('--model_path', default='models/model__epoch1/', type=str, required=False, help='')
        parser.add_argument('--data_path', default='datas/', type=str, required=False, help='')
        parser.add_argument('--cache_path', default='caches/', type=str, required=False, help='')
        parser.add_argument('--vocab_fname', default='models/vocab.txt', type=str, required=False, help='')

        # TODO: let commandline temp args override args in config_file
        args = parser.parse_args()
        if args.config_file != '':
            parser.set_defaults(**yaml.load(open(args.config_file)))
            args = parser.parse_args()

        return args

    def build_vocab(self):
        pass
    
    def build_dataloaders(self):
        pass
    
    def build_model(self):
        pass

    def build_metric_fns(self):
        pass

    def run(self):
        pass

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.model_path), strict=False)


if __name__ == '__main__':
    evaler = Evaler()
    evaler.run()
