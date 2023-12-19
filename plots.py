import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def compare_behavior_vs_prediction(behavior, prediction, save_path=None, plot_dprime=True):
    plt.figure(figsize=(4, 4))
    plt.scatter(behavior[:: 2], prediction[:: 2], s=10, color='red')
    plt.scatter(behavior[1:: 2], prediction[1:: 2], s=10, color='blue')
    
    plt.plot([0, 1], [0, 1], color='gray')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.xlabel('Behavior')
    plt.ylabel('Prediction')

    # show correlation
    behavior = np.array(behavior)
    prediction = np.array(prediction)
    corr = np.corrcoef(behavior, prediction)[0, 1]
    spearman = stats.spearmanr(behavior, prediction)
    plt.title(f'Pearson: {corr:.2f}, Spearman: {spearman[0]:.2f}')
    plt.subplots_adjust(left=0.2, bottom=0.2)

    if save_path:
        plt.savefig(os.path.join(save_path, 'beh_vs_pred.png'))
    else:
        plt.show()

    if plot_dprime:
        plt.figure(figsize=(4, 4))
        false_alarm = 1 - np.array(behavior[:: 2]), 1 - np.array(prediction[:: 2])
        hit = 1 - np.array(behavior[1:: 2]), 1 - np.array(prediction[1:: 2])
        
        false_alarm = np.clip(false_alarm, 1e-3, 1 - 1e-3)
        hit = np.clip(hit, 1e-3, 1 - 1e-3)
        d_prime = stats.norm.ppf(hit) - stats.norm.ppf(false_alarm)
        
        plt.scatter(d_prime[0], d_prime[1], s=10)
        plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.plot([0, 5], [0, 5], color='gray')

        plt.xlabel('Behavior (d\')')
        plt.ylabel('Prediction (d\')')

        # show correlation
        corr = np.corrcoef(d_prime[0], d_prime[1])[0, 1]
        spearman = stats.spearmanr(d_prime[0], d_prime[1])
        plt.title(f'Pearson: {corr:.2f}, Spearman: {spearman[0]:.2f}')

        if save_path:
            plt.savefig(os.path.join(save_path, 'beh_vs_pred_dprime.png'))
        else:
            plt.show()

        return d_prime    

def compare_behavior_vs_distance(behavior, distance, save_path=None):

    plt.figure(figsize=(4, 4))
    plt.scatter(behavior, distance, s=10)
    
    plt.xlabel('Behavior')
    plt.ylabel('Distance')

    # show correlation
    behavior = np.array(behavior)
    distance = np.array(distance)
    corr = np.corrcoef(behavior, distance)[0, 1]
    spearman = stats.spearmanr(behavior, distance)
    plt.title(f'Pearson: {corr:.2f}, Spearman: {spearman[0]:.2f}')
    plt.subplots_adjust(left=0.2, bottom=0.2)

    if save_path:
        plt.savefig(os.path.join(save_path, 'beh_vs_dist.png'))
    else:
        plt.show()

def compare_acc_vs_set_size(set_sizes, accs, y_label, ylim=None, save_path=None, labels=None):
    plt.figure(figsize=(4, 3.5))
    
    if labels is None:
        plt.plot(set_sizes, accs)
    else:
        for i in range(len(accs)):
            plt.plot(set_sizes, accs[i], label=labels[i])
        plt.legend()
    
    plt.xlabel('Set Size')
    plt.ylabel(y_label)
    plt.xticks([4, 8, 12])

    # show correlation
    set_sizes = np.array(set_sizes)
    accs = np.array(accs)

    # increase the margin
    plt.subplots_adjust(left=0.2, bottom=0.2)
    if ylim is not None:
        plt.ylim(ylim)

    # do not show the top and right margin
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(os.path.join(save_path, f'{y_label}_vs_set_size.png'))
    else:
        plt.show()