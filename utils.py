'''
Utility classes/functions for dealing with pretrained models.
'''

import os
import torch
import collections

from models import RNN, GRU


class Experiment():
    '''
    The experiment-run directory produced by ptb-lm.py during model training.
    '''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.params_path = os.path.join(dir_path, 'best_params.pt')
        self.config_path = os.path.join(dir_path, 'exp_config.txt')
        self.lc_path = os.path.join(dir_path, 'learning_curves.npy')
        self.log_path = os.path.join(dir_path, 'log.txt')
        self.config = ExpConfig(self.config_path).config_dict

    def load_model(self):
        # Magic number is the vocab size - always 10000
        if self.config['model'] == 'RNN':
            model = RNN(self.config['emb_size'], self.config['hidden_size'], self.config['seq_len'],
                        self.config['batch_size'], 10000, self.config['num_layers'],
                        self.config['dp_keep_prob'])
        elif self.config['model'] == 'GRU':
            model = GRU(self.config['emb_size'], self.config['hidden_size'], self.config['seq_len'],
                        self.config['batch_size'], 10000, self.config['num_layers'],
                        self.config['dp_keep_prob'])

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(self.params_path))
        else:
            model.load_state_dict(torch.load(self.params_path, map_location='cpu'))
        model.eval()
        return model


class ExpConfig():
    '''
    Parses config_path into meaningful structured variables.
    '''

    def __init__(self, config_path):
        self.config_path = config_path
        self.config_dict = self.parse()

    def parse(self):
        with open(self.config_path, 'r') as fp:
            lines = fp.readlines()
        config = {}
        for line in lines:
            split = line.split()
            # Cast number types
            if split[1].isdigit():
                split[1] = int(split[1])
            elif '.' in split[1] and split[1].replace('.', '').isdigit():
                split[1] = float(split[1])
            config[split[0]] = split[1]
        return config


def prepare_data():
    # LOAD DATA
    print('Loading data...')
    raw_data = ptb_raw_data(data_path='data')
    _, _, _, word_to_id, id_2_word = raw_data   # We just need these
    print('     Data loaded.')
    return word_to_id, id_2_word


def ptb_raw_data(data_path=None, prefix="ptb"):
    # Processes the raw data from text files
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def save_gradient(loss, hiddens):
    """
    tired friday coding, we can compute the grad and save it to a numpy npz
    !Not tested version!
    :param loss:
    :param hiddens:
    :return:
    """
    grad = torch.autograd.grad(loss, hiddens.detach().cpu(), retain_graph=True)
    grad_res = grad.numpy()
    np.save(os.path.join('.', 'avg_5_2.npy'), grad_res)