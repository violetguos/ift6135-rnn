'''
File for 5.2
For one minibatch of training data, compute the average gradient
of the loss at the final timestep with respect to the hidden
state at each timestep.
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from utils import Experiment, prepare_data, repackage_hidden, ptb_iterator, save_gradient

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compute average loss gradient at final timestep with respect to hidden states at each timestep.')
    parser.add_argument('exp_dir', type=str,
                        help='The directory from the experiment run for the model you want.')
    args = parser.parse_args()

    # Use the GPU if you have one
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set up the model
    print('Loading model...')
    exp = Experiment(args.exp_dir)
    model = exp.load_model()
    model = model.to(device)
    print('     Model loaded.')

    batch_size = model.batch_size
    seq_len = model.seq_len

    # Set up the training data
    train_data, _, _, _ = prepare_data()

    # Set up loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize hidden state
    hidden = model.init_hidden()
    hidden = hidden.to(device)

    # One minibatch of training data
    for step, (x, y) in enumerate(ptb_iterator(train_data, batch_size, seq_len)):
        if step == 1:
            break

        # Pass through network
        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).contiguous().to(device)
        tt = torch.squeeze(targets.view(-1, batch_size * seq_len))

        # Calculate loss
        loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)

        # Now backprop
        loss.backward(retain_graph=True)

        # Get hidden gradients
        hidden_dict = model.hidden_dict
        mean_grads = []
        for t in range(34):     #range(1):
            print("time step {}".format(t))
            to_concat = []
            grad = torch.autograd.grad(loss, hidden_dict[t], retain_graph=True)

            # On David's suggestion
            # concatenate the gradients for all hidden layers (at each timestep)
            grad = torch.cat(grad, dim=0)
            mean_grads.append(grad.mean(0))
        mean_grads = torch.stack(mean_grads)

        # Take Euclidean norm
        normed_grads = mean_grads.norm(p=2, dim=-1)

        # Rescale the values of each curve to [0, 1] (Tegan's formulation)
        scaled_grads = (normed_grads - normed_grads.min()) / \
                         (normed_grads.max() - normed_grads.min())

        # Log/save
        print(scaled_grads)
        np.save(os.path.join('.', 'avg_5_2_{}.npy'.format(exp.config['model'])), scaled_grads)
