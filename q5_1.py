'''
File for 5.1
Computing the average loss at each timestep over the whole validation set.
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from utils import Experiment, prepare_data, repackage_hidden, ptb_iterator

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compute average loss over validation for a given model.')
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
    num_batches = 0

    # Set up the validation data
    _, valid_data, _, _ = prepare_data()

    # Set up loss function
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Initialize hidden state
    hidden = model.init_hidden()
    hidden = hidden.to(device)

    # Loop through validation set (one epoch)
    mean_losses = torch.zeros(seq_len, device=device)
    for step, (x, y) in enumerate(ptb_iterator(valid_data, batch_size, seq_len)):
        print('Step {}'.format(step))
        num_batches += 1

        # On David's recommendation
        hidden = model.init_hidden()

        # Pass through network
        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).contiguous().to(device)
        tt = torch.squeeze(targets)

        # Calculate loss
        # Collect per-timestep losses from logits across all examples
        loss = loss_fn(outputs.contiguous().permute(1, 2, 0), tt)
        # Take mean "down the column" of all t=1, t=2, ...
        mean_loss = loss.detach().mean(0)
        # Keep running sum
        mean_losses += mean_loss

    # On David's formulation
    final_losses = (mean_losses / num_batches).detach().cpu().numpy()

    # Log/save
    np.save(os.path.join('.', 'avg_5_1.npy'), final_losses)
