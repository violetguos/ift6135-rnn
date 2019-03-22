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

from utils import Experiment, prepare_data, repackage_hidden, ptb_iterator

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

        # We want to keep track of the gradients at each hidden layer,
        # over each timestep. So install hooks to the hidden layers
        def register_grad_hook(tensor):
            def hook(grad):
                tensor.hidden_grad = grad
            tensor.register_backward_hook(hook)

        for hidden in model.hiddens:
            register_grad_hook(hidden)

        # Calculate loss
        loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)

        # Now backprop
        loss.backward()

        # Collect per-timestep loss wrt hidden states, of all t=1, t=2, ...
        mean_losses = torch.stack(model.hiddens).mean(1)

        # Take Euclidean norm
        normed_losses = mean_losses.norm(p=2, dim=-1)

        # Rescale the values of each curve to [0, 1] (Tegan's formulation)
        scaled_losses = (normed_losses - normed_losses.min()) / \
                         (normed_losses.max() - normed_losses.min())

        # Log/save
        np.save(os.path.join('.', 'avg_5_2.npy'), scaled_losses)
