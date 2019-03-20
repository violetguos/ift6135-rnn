'''
A script for generating (mostly) loss plots.
Run it from the root of the repo.
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def get_walltimes(log_path):
    # Walltime is the cumulative sum of time each epoch takes
    with open(log_path, 'r') as fp:
        log = fp.readlines()
    epoch_times = [float(line.split('\t')[-1].split(':')[-1]) for line in log]
    walltimes = []
    for i, time in enumerate(epoch_times):
        if i == 0:
            walltimes.append(time)
            prev_time = time
        else:
            time = time + prev_time
            walltimes.append(time)
    return walltimes


def gen_train_valid_ppl_epoch(x, save_dir):
    train_ppls = x['train_ppls']
    valid_ppls = x['val_ppls']
    epochs = range(len(train_ppls))
    plt.plot(epochs, train_ppls, color='red', label='Training PPL')
    plt.plot(epochs, valid_ppls, color='blue', label='Validation PPL')
    plt.title('Perplexity over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity (PPL)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_valid_ppl_epoch.png'))
    plt.clf()   # Clear


def gen_train_valid_ppl_walltime(x, walltimes, save_dir):
    train_ppls = x['train_ppls']
    valid_ppls = x['val_ppls']
    plt.plot(walltimes, train_ppls, color='red', label='Training PPL')
    plt.plot(walltimes, valid_ppls, color='blue', label='Validation PPL')
    plt.title('Perplexity over wall-clock-time')
    plt.xlabel('Wall-clock-time (s)')
    plt.ylabel('Perplexity (PPL)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_valid_ppl_walltime.png'))
    plt.clf()   # Clear


def gen_valid_arch_ppl_epoch(arch_ppls, save_dir):
    for arch, xs in arch_ppls.items():
        for x in xs:
            valid_ppls = x['val_ppls']
            epochs = range(len(valid_ppls))
            plt.plot(epochs, valid_ppls, color='blue')
        plt.title('Perplexity over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity (PPL)')
        plt.savefig(os.path.join(save_dir, '{}_ppl_epoch.png'.format(arch)))
        plt.clf()   # Clear


def gen_valid_arch_ppl_walltime(arch_ppls, arch_walltimes, save_dir):
    for arch, xs in arch_ppls.items():
        for i, x in enumerate(xs):
            valid_ppls = x['val_ppls']
            walltimes = arch_walltimes[arch][i]
            plt.plot(walltimes, valid_ppls, color='blue')
        plt.title('Perplexity over wall-clock-time')
        plt.xlabel('Wall-clock-time (s)')
        plt.ylabel('Perplexity (PPL)')
        plt.savefig(os.path.join(save_dir, '{}_ppl_epoch.png'.format(arch)))
        plt.clf()   # Clear


def gen_valid_opt_ppl_epoch(opt_ppls, save_dir):
    for opt, xs in opt_ppls.items():
        for x in xs:
            valid_ppls = x['val_ppls']
            epochs = range(len(valid_ppls))
            plt.plot(epochs, valid_ppls, color='blue')
        plt.title('Perplexity over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity (PPL)')
        plt.savefig(os.path.join(save_dir, '{}_ppl_epoch.png'.format(opt)))
        plt.clf()   # Clear


def gen_valid_opt_ppl_walltime(opt_ppls, opt_walltimes, save_dir):
    for opt, xs in opt_ppls.items():
        for i, x in enumerate(xs):
            valid_ppls = x['val_ppls']
            walltimes = opt_walltimes[arch][i]
            plt.plot(walltimes, valid_ppls, color='blue')
        plt.title('Perplexity over wall-clock-time')
        plt.xlabel('Wall-clock-time (s)')
        plt.ylabel('Perplexity (PPL)')
        plt.savefig(os.path.join(save_dir, '{}_ppl_epoch.png'.format(opt)))
        plt.clf()   # Clear


if __name__ == '__main__':
    # Parse arguments
    tasks = ['epochs', 'walltime', 'architecture', 'optimizer']
    parser = argparse.ArgumentParser(description='Generate different types of plots.')
    parser.add_argument('task', type=str,
                        help='The plots you want to generate. Possible values are: {}'.format(tasks))
    args = parser.parse_args()

    # For collection
    arch_ppls, arch_walltimes = {}, {}
    opt_ppls, opt_walltimes = {}, {}

    # Walk through experiment result directories
    subdirs = next(os.walk('.'))[1]
    for subdir in subdirs:
        if 'exp' in subdir:

            # Generate training + validation PPL plots over epochs for all experiments
            if args.task == 'epochs':
                lc_path = os.path.join(subdir, 'learning_curves.npy')
                x = np.load(lc_path)[()]
                gen_train_valid_ppl_epoch(x, subdir)

            # Generate training + validation PPL plots over walltime for all experiments
            elif args.task == 'walltime':
                lc_path = os.path.join(subdir, 'learning_curves.npy')
                x = np.load(lc_path)[()]
                # Get walltimes
                log_path = os.path.join(subdir, 'log.txt')
                walltimes = get_walltimes(log_path)
                # Get plots
                gen_train_valid_ppl_walltime(x, walltimes, subdir)

            # Generate one validation PPL plot over epochs and walltime over all experiments per each architecture.
            # Do this by first accumulating the data for each arch, then do the plot outside this main if statement.
            elif args.task == 'architecture':
                # Collect validation ppls
                config_path = os.path.join(subdir, 'exp_config.txt')
                with open(config_path, 'r') as fp:
                    config = fp.readlines()
                arch = config[9].split('    ')[-1].strip().lower()    # This is the line where "model" is in the config
                print('arch:', arch)
                lc_path = os.path.join(subdir, 'learning_curves.npy')
                x = np.load(lc_path)[()]
                if arch not in arch_ppls:
                    arch_ppls[arch] = [x]
                else:
                    arch_ppls[arch].append(x)

                # Collect walltimes
                log_path = os.path.join(subdir, 'log.txt')
                walltimes = get_walltimes(log_path)
                if arch not in arch_walltimes:
                    arch_walltimes[arch] = [walltimes]
                else:
                    arch_walltimes[arch].append(walltimes)

            # Generate one validation PPL plot over epochs and walltime over all experiments per each optimizer.
            # Do this by first accumulating the data for each arch, then do the plot outside this main if statement.
            elif args.task == 'optimizer':
                # Collect validation ppls
                config_path = os.path.join(subdir, 'exp_config.txt')
                with open(config_path, 'r') as fp:
                    config = fp.readlines()
                opt = config[12].split('    ')[-1].split('\t').strip().lower()    # This is the line where "optimizer" is in the config
                lc_path = os.path.join(subdir, 'learning_curves.npy')
                x = np.load(lc_path)[()]
                if opt not in opt_ppls:
                    opt_ppls[arch] = x
                else:
                    opt_ppls[arch].append(x)

                # Collect walltimes
                log_path = os.path.join(subdir, 'log.txt')
                walltimes = get_walltimes(log_path)
                if opt not in opt_walltimes:
                    opt_walltimes[opt] = [walltimes]
                else:
                    opt_walltimes[opt].append(walltimes)

    # Generate plots for architecture
    if args.task == 'architecture':
        gen_valid_arch_ppl_epoch(arch_ppls, '.')
        gen_valid_arch_ppl_walltime(arch_ppls, arch_walltimes, '.')

    # Generate plots for optimizer
    elif args.task == 'optimizer':
        gen_valid_opt_ppl_epoch(opt_ppls, '.')
        gen_valid_opt_ppl_epoch(opt_ppls, opt_walltimes, '.')
