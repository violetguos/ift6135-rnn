'''
Plot the average losses from q5_1.py
'''

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average loss over validation for a given model.')
    parser.add_argument('plot_name', type=str,
                        help='name of the plot!')
    args = parser.parse_args()

    # Load
    x = np.load(os.path.join('.', 'avg_5_2.npy'))

    # Plot
    timesteps = range(35)
    plt.plot(timesteps, x)
    plt.title('Average per-timestep loss over validation set')
    plt.xlabel('Timestep')
    plt.ylabel('Average loss')

    # Save
    fname = args.plot_name + '.png'
    plot_path = os.path.join('plots', fname)
    plt.savefig(plot_path)
    plt.clf()
