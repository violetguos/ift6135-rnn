'''
Plot the average losses from q5_1.py
'''

import matplotlib.pyplot as plt
import os
import numpy as np


if __name__ == '__main__':
    # Load
    x = np.load(os.path.join('.', 'avg_5_1.npy'))

    # Plot
    timesteps = range(35)
    plt.plot(timesteps, x)
    plt.title('Average per-timestep loss over validation set')
    plt.xlabel('Timestep')
    plt.ylabel('Average loss')

    # Save
    plot_path = os.path.join('plots', 'avg_5_1.png')
    plt.savefig(plot_path)
    plt.clf()
