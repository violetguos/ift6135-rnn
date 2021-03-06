'''
Plot the average losses from q5_1.py
'''

import matplotlib.pyplot as plt
import os
import numpy as np


if __name__ == '__main__':
    # Load
    x1 = np.load(os.path.join('.', 'avg_5_2_RNN.npy'))
    x2 = np.load(os.path.join('.', 'avg_5_2_GRU.npy'))

    # Plot
    timesteps = range(34) #(!) magic number cuz we aint got no time
    plt.plot(timesteps, x1, label='Best RNN')
    plt.plot(timesteps, x2, label='Best GRU')
    plt.title('Norm of loss gradients at final time step over time')
    plt.xlabel('Timestep')
    plt.ylabel('Norm of loss gradient')
    plt.legend()

    # Save
    plot_path = os.path.join('plots', 'avg_5_2.png')
    plt.savefig(plot_path)
    plt.clf()
