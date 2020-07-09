
import os
import time
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'
PRESET_SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]
EN_TO_ZH = dict(
    location='地址',
    name='姓名',
    weight='体重',
    gender='性别',
    age='年龄',
    constellation='星座',
    hobby='爱好',
    speciality='特长',
) 


class Vocab:
    def __init__(
        self,
        vocab,
        profiles,
        data_path,
        special_tokens=None
    ):
        self.stoi_map = {}
        self.itos_map = {}
        self.profile_stoi_map = {}
        self.profile_itos_map = {}
        self.binary_lable = dict(
            positive=1,
            negative=0,
        )

        if special_tokens is None:
            special_tokens = PRESET_SPECIAL_TOKENS
        else:
            special_tokens = PRESET_SPECIAL_TOKENS + special_tokens

        for i, (k, v) in enumerate(profiles.items()):
            self.profile_stoi_map[k] = (i, v)
            self.profile_itos_map[i] = (k, v)

        if vocab is None:
            self.__init(data_path, special_tokens)
            return

        for k, v in vocab.items():
            self.stoi_map[k] = (v.index, v.count)
            self.itos_map[v.index] = k

        i = len(self.stoi_map)
        for k in special_tokens:
            self.stoi_map[k] = [i, 1000]
            self.itos_map[i] = k
            i += 1

    def __init(
        self,
        data_path,
        special_tokens=None
    ):
        examples = get_examples(data_path, 'train')

        self.stoi_map = {}
        self.itos_map = {}

        i = 0
        for post, resp, _ in examples:
            for k in set(post + resp):
                if k not in self.stoi_map:
                    self.stoi_map[k] = [i, 0]
                    i += 1
                self.stoi_map[k][1] += 1

        i = len(self.stoi_map)
        for k in special_tokens:
            self.stoi_map[k] = [i, 1000]
            i += 1

        self.itos_map = {i: k for k, (i, _) in self.stoi_map.items()}

    def __len__(self):
        return len(self.stoi_map)

    def stoi(self, s):
        return self.stoi_map.get(s, self.stoi_map[UNK])[0]

    def itos(self, i):
        return self.itos_map[i]

    def exists_profile(self, s):
        return s in EN_TO_ZH and EN_TO_ZH[s] in self.profile_stoi_map \
                or s in self.profile_stoi_map

    def profile_stoi(self, s):
        if s in EN_TO_ZH:
            s = EN_TO_ZH[s]
        return self.profile_stoi_map[s][0]

    def profile_itos(self, i):
        return self.profile_itos_map[i]

    def binary_stoi(self, s):
        return self.binary_lable[s]

    def binary_itos(self, i):
        return [k for k, v in self.binary_lable.items() if i == v][0]


class PersonaDataset(Dataset):
    def __init__(
        self,
        vocab,
        max_seq_length,
        data_path,
        cache_path,
        limit_length=None,
        mode='train',
        overwrite_cache=True,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_path,
            "cached_{}_{}".format(
                mode, str(max_seq_length),
            ),
        )
        
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                print(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                print(f"Creating features from dataset file at {data_path}")

                examples = get_examples(data_path, mode)
                if limit_length is not None:
                    examples = examples[:limit_length]
                    
                self.features = convert_examples_to_features(
                    vocab,
                    examples,
                    max_length=max_seq_length,
                    mode=mode,
                )
                start = time.time()
                # torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                print("Saving features into cached file %s [took %.3f s]" % (cached_features_file, time.time() - start))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


def get_examples(path, mode):
    post_file_path = os.path.join(path, mode + '.post')
    resp_file_path = os.path.join(path, mode + '.resp')
    key_file_path = os.path.join(path, mode + '.keys')

    with open(post_file_path) as f:
        posts = f.read()
    with open(resp_file_path) as f:
        resps = f.read()

    keys = ''
    if os.path.exists(key_file_path):
        with open(key_file_path) as f:
            keys = f.read()

    def parse(s):
        if len(s) == 0:
            return []
        # last row is blank
        return list(map(lambda x: x.split(), s.split('\n')))[:-1]

    posts_arr = parse(posts)
    resps_arr = parse(resps)
    if keys == '':
        # placehold for early_stage_train
        keys_arr = [['negative', 'name']] * len(posts_arr)
    else:
        keys_arr = parse(keys)

    return list(zip(posts_arr, resps_arr, keys_arr))


def convert_examples_to_features(
    vocab,
    examples,
    max_length,
    mode
):
    ret = []
    for post, resp, key in examples:
        if not vocab.exists_profile(key[1]):
            continue
        ipost = [vocab.stoi(k) for k in post[:max_length]] + [vocab.stoi(EOS)]
        iresp = [vocab.stoi(SOS)] + [vocab.stoi(k) for k in resp[:max_length]] + [vocab.stoi(EOS)]
        ikey = [vocab.binary_stoi(key[0]), vocab.profile_stoi(key[1])]
        ret.append((ipost, iresp, ikey))
    return ret


def convert_profiles_to_features(
    vocab,
    profiles
):
    return [[vocab.stoi(k), vocab.stoi(v)] 
            for k, v in profiles.items()]


# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html?highlight=collate_fn
def generate_batch(batch, pad_idx):
    post, resp, key = zip(*batch)
    post_lens = [len(v) for v in post]
    resp_lens = [len(v) for v in resp]

    fn = lambda x: list(map(torch.tensor, x)) 
    post_pad = pad_sequence(fn(post), padding_value=pad_idx)
    resp_pad = pad_sequence(fn(resp), padding_value=pad_idx)
    key = torch.tensor(key)

    return post_pad, resp_pad, post_lens, resp_lens, key


def retokenize(fname):
    import jieba
    res = []
    with open(fname) as f:
        ss = f.read()
        for line in ss.split('\n'):
            line = line.replace(' ', '')
            line = list(jieba.cut(line))
            line = ' '.join(line)
            res.append(line)

    name, ext = os.path.splitext(fname) 
    fname_res = name + '_retoken' + ext
    with open(fname_res, 'w') as f:
        ss = '\n'.join(res)
        f.write(ss)

