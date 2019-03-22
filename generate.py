'''
File for generating new sequences (Problem 5.3).
Generated sequences are saved under generated_sequences/.
'''

import argparse
import os
import numpy as np
import torch

from utils import Experiment, prepare_data


def log_sequences(sequences, arch, num_sequences, length, sequence_dir):
    file_name = '{}_{}_sequences_length_{}.txt'.format(arch, num_sequences, length)
    file_path = os.path.join(sequence_dir, file_name)
    with open(file_path, 'w') as fp:
        for sequence in sequences:
            fp.write(sequence)
            fp.write('\n')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate new sequences from a trained model.')
    parser.add_argument('exp_dir', type=str,
                        help='The directory from the experiment run for the model you want to generate from.')
    parser.add_argument('num_sequences', type=int,
                        help='The number of sequences you want to generate.')
    parser.add_argument('length', type=int,
                        help='The length of the sequence you want to generate.')
    args = parser.parse_args()
    sequence_dir = os.path.join('.', 'generated_sequences')

    # Use the GPU if you have one
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set up the model
    print('Loading model...')
    exp = Experiment(args.exp_dir)
    model = exp.load_model()
    print('     Model loaded.')

    # Set up the data (to be able to sample initial words)
    _, _, word_to_id, id_2_word = prepare_data()

    # Initialize hidden state
    hidden = model.init_hidden()
    hidden = hidden.to(device)

    # Generate num_sequences sequences
    one_inp = word_to_id['<eos>']   # Seed with <eos> token
    inp = torch.from_numpy(np.array([one_inp for i in range(args.num_sequences)]))
    # print("size inp", inp.size())
    samples = model.generate(inp, hidden, args.length)
    # necessary
    samples = samples.transpose(0, 1)

    # Separate out the sequences and build sentences from them
    sequences = []
    for sample in samples:
        # print('sample size', sample.size())

        words = [id_2_word[i.item()] for i in sample]
        sequence = ' '.join(words)
        print(sequence)
        sequences.append(sequence)

    # Save to log
    log_sequences(sequences, exp.config['model'], args.num_sequences, args.length, sequence_dir)
    print('Sequences saved.')
