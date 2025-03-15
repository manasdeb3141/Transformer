
import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

def get_top_N_values(input_vals, N):
    # Get the indices of the N largest elements
    indices = np.argpartition(input_vals, N)[-N:]

    # Return the N largest elements and their indices
    return input_vals[indices], indices

def  plot_JSD_lineplot(JSD_stats, epoch_list, attention_layer):
    query_JSD_min_list = [JSD_stat["query_JSD_min"] for JSD_stat in JSD_stats]
    query_JSD_max_list = [JSD_stat["query_JSD_max"] for JSD_stat in JSD_stats]
    key_JSD_min_list = [JSD_stat["key_JSD_min"] for JSD_stat in JSD_stats]
    key_JSD_max_list = [JSD_stat["key_JSD_max"] for JSD_stat in JSD_stats]
    value_JSD_min_list = [JSD_stat["value_JSD_min"] for JSD_stat in JSD_stats]
    value_JSD_max_list = [JSD_stat["value_JSD_max"] for JSD_stat in JSD_stats]

    mean_query_JSD_max = np.mean(query_JSD_max_list)
    std_query_JSD_max = np.std(query_JSD_max_list)
    mean_key_JSD_max = np.mean(key_JSD_max_list)
    std_key_JSD_max = np.std(key_JSD_max_list)
    mean_value_JSD_max = np.mean(value_JSD_max_list)
    std_value_JSD_max = np.std(value_JSD_max_list)

    sns.set_theme()
    plt.plot(epoch_list, query_JSD_max_list, label="Q' max JSD", color='b', linestyle='-')
    # plt.fill_between(epoch_list, query_JSD_max_list-std_query_JSD_max, query_JSD_max_list+std_query_JSD_max, color='blue', alpha=0.3)
    plt.plot(epoch_list, key_JSD_max_list, label="K' max JSD", color='r', linestyle='-')
    # plt.fill_between(epoch_list, key_JSD_max_list-std_key_JSD_max, key_JSD_max_list+std_key_JSD_max, color='red', alpha=0.3)
    plt.xticks(range(0, 20), range(0, 20))
    plt.legend()
    plt.margins(x=0, y=0)
    plt.title(f'Maximum Jensen-Shannon Divergence for attention layer {attention_layer}')
    plt.xlabel('Epoch')
    plt.ylabel('Jensen-Shannon Divergence')
    plt.show()



def  plot_JSD_estimate(JSD_stats, epoch_list, attention_layer):
    fig, axs = plt.subplots(3,3)

    query_JSD_min_list = [JSD_stat["query_JSD_min"] for JSD_stat in JSD_stats]
    query_JSD_max_list = [JSD_stat["query_JSD_max"] for JSD_stat in JSD_stats]
    query_JSD_mean_list = [JSD_stat["query_JSD_mean"] for JSD_stat in JSD_stats]
    key_JSD_min_list = [JSD_stat["key_JSD_min"] for JSD_stat in JSD_stats]
    key_JSD_max_list = [JSD_stat["key_JSD_max"] for JSD_stat in JSD_stats]
    key_JSD_mean_list = [JSD_stat["key_JSD_mean"] for JSD_stat in JSD_stats]
    value_JSD_min_list = [JSD_stat["value_JSD_min"] for JSD_stat in JSD_stats]
    value_JSD_max_list = [JSD_stat["value_JSD_max"] for JSD_stat in JSD_stats]
    value_JSD_mean_list = [JSD_stat["value_JSD_mean"] for JSD_stat in JSD_stats]

    # Plot the Q' JSD stats
    axs[0, 0].bar(epoch_list, height=query_JSD_min_list, color='blue')
    axs[0, 0].set_title(f"Query JSD Min")
    max_val = max(query_JSD_min_list)
    axs[0, 0].set_ylim([0, max_val+0.1*max_val])
    axs[0, 1].bar(epoch_list, height=query_JSD_max_list, color='blue')
    axs[0, 1].set_title(f"Query JSD Max")
    axs[0, 2].bar(epoch_list, height=query_JSD_mean_list, color='blue')
    axs[0, 2].set_title(f"Query JSD Mean")

    axs[1, 0].bar(epoch_list, height=key_JSD_min_list, color='blue')
    axs[1, 0].set_title(f"Key JSD Min")
    max_val = max(key_JSD_min_list)
    axs[1, 0].set_ylim([0, max_val+0.1*max_val])
    axs[1, 1].bar(epoch_list, height=key_JSD_max_list, color='blue')
    axs[1, 1].set_title(f"Key JSD Max")
    axs[1, 2].bar(epoch_list, height=key_JSD_mean_list, color='blue')
    axs[1, 2].set_title(f"Key JSD Mean")

    axs[2, 0].bar(epoch_list, height=value_JSD_min_list, color='blue')
    axs[2, 0].set_title(f"Value JSD Min")
    max_val = max(value_JSD_min_list)
    axs[2, 0].set_ylim([0, max_val+0.1*max_val])
    axs[2, 1].bar(epoch_list, height=value_JSD_max_list, color='blue')
    axs[2, 1].set_title(f"Value JSD Max")
    axs[2, 2].bar(epoch_list, height=value_JSD_mean_list, color='blue')
    axs[2, 2].set_title(f"Value JSD Mean")

    fig.suptitle(f"Jensen-Shannon Divergence Estimates for Attention Layer {attention_layer}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

def plot_JSD_bars(QKV_dict, word_list):
    query_JSD_estimate = QKV_dict["query_JSD_estimate"]
    key_JSD_estimate = QKV_dict["key_JSD_estimate"]
    value_JSD_estimate = QKV_dict["value_JSD_estimate"]
    orig_input_words = QKV_dict["input_words"]

    np.fill_diagonal(query_JSD_estimate, 0)
    np.fill_diagonal(key_JSD_estimate, 0)
    np.fill_diagonal(value_JSD_estimate, 0)

    for word in word_list:
        word_index = orig_input_words.index(word)
        query_JSD = query_JSD_estimate[word_index]
        key_JSD = key_JSD_estimate[word_index]
        value_JSD = value_JSD_estimate[word_index]

        # Remove the word for which we are plotting the JSD,
        # from the list
        input_words = orig_input_words.copy()
        input_words.pop(word_index)
        query_JSD = np.delete(query_JSD, word_index)
        key_JSD = np.delete(key_JSD, word_index)
        value_JSD = np.delete(value_JSD, word_index)

        fig, axs = plt.subplots(3,1)
        x_vals = np.arange(0, len(input_words))

        # Plot Q'
        top_vals, N_top_indices = get_top_N_values(query_JSD, 3)
        top_indices = np.argwhere(query_JSD >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_Q = axs[0].bar(x_vals, query_JSD, color=bar_colors)
        axs[0].set_title(f"Q' Jensen-Shannon Divergence")
        axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[0].set_ylabel("Jensen-Shannon Divergence")
        for bar in barplot_Q:
            height = bar.get_height()
            axs[0].text(bar.get_x() + bar.get_width()/2., 0.80*height, f'{height:.2f}', ha='center', va='bottom')

        # Plot K'
        top_vals, N_top_indices = get_top_N_values(key_JSD, 3)
        top_indices = np.argwhere(key_JSD >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_K = axs[1].bar(x_vals, key_JSD, color=bar_colors)
        axs[1].set_title(f"K' Jensen-Shannon Divergence")
        axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[1].set_ylabel("Jensen-Shannon Divergence")
        for bar in barplot_K:
            height = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width()/2., 0.80*height, f'{height:.2f}', ha='center', va='bottom')

        # Plot V'
        top_vals, N_top_indices = get_top_N_values(value_JSD, 3)
        top_indices = np.argwhere(value_JSD >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_V = axs[2].bar(x_vals, value_JSD, color=bar_colors)
        axs[2].set_title(f"V' Jensen-Shannon Divergence")
        axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[2].set_ylabel("Jensen-Shannon Divergence")
        for bar in barplot_V:
            height = bar.get_height()
            axs[2].text(bar.get_x() + bar.get_width()/2., 0.80*height, f'{height:.2f}', ha='center', va='bottom')

        fig.suptitle(f"Jensen-Shannon divergence between the word '{word}' and other words")
        plt.subplots_adjust(hspace=0.5)
        plt.show(block=True)

def plot_MI_bars(QKV_dict, word_list):
    query_MI_estimate = QKV_dict["query_MI_estimate"]
    key_MI_estimate = QKV_dict["key_MI_estimate"]
    value_MI_estimate = QKV_dict["value_MI_estimate"]
    orig_input_words = QKV_dict["input_words"]

    np.fill_diagonal(query_MI_estimate, 0)
    np.fill_diagonal(key_MI_estimate, 0)
    np.fill_diagonal(value_MI_estimate, 0)

    for word in word_list:
        word_index = orig_input_words.index(word)
        query_MI = query_MI_estimate[word_index]
        key_MI = key_MI_estimate[word_index]
        value_MI = value_MI_estimate[word_index]

        # Remove the word for which we are plotting the MI
        # from the list
        input_words = orig_input_words.copy()
        input_words.pop(word_index)
        query_MI = np.delete(query_MI, word_index)
        key_MI = np.delete(key_MI, word_index)
        value_MI = np.delete(value_MI, word_index)

        fig, axs = plt.subplots(3,1)
        x_vals = np.arange(0, len(input_words))

        # Plot Q'
        top_vals, N_top_indices = get_top_N_values(query_MI, 3)
        top_indices = np.argwhere(query_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_Q = axs[0].bar(x_vals, query_MI, color=bar_colors)
        axs[0].set_title(f"Q' Mutual Information")
        axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[0].set_ylabel("Mutual Information")
        for bar in barplot_Q:
            height = bar.get_height()
            axs[0].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        for sp in ['top', 'right']:
            axs[0].spines[sp].set_visible(False)

        # Plot K'
        top_vals, N_top_indices = get_top_N_values(key_MI, 3)
        top_indices = np.argwhere(key_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_K = axs[1].bar(x_vals, key_MI, color=bar_colors)
        axs[1].set_title(f"K' Mutual Information")
        axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[1].set_ylabel("Mutual Information")
        for bar in barplot_K:
            height = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        for sp in ['top', 'right']:
            axs[1].spines[sp].set_visible(False)

        # Plot V'
        top_vals, N_top_indices = get_top_N_values(value_MI, 3)
        top_indices = np.argwhere(value_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_V = axs[2].bar(x_vals, value_MI, color=bar_colors)
        axs[2].set_title(f"V' Mutual Information")
        axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[2].set_ylabel("Mutual Information")
        for bar in barplot_V:
            height = bar.get_height()
            axs[2].text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        for sp in ['top', 'right']:
            axs[2].spines[sp].set_visible(False)

        fig.suptitle(f"Mutual Information between the word '{word}' and other words")
        plt.subplots_adjust(hspace=0.5)
        plt.show(block=True)

def plot_QKV_prime(QKV_dict, epoch, attention_layer, sentence_id):
    query_MI_estimate = QKV_dict["query_MI_estimate"]
    key_MI_estimate = QKV_dict["key_MI_estimate"]
    value_MI_estimate = QKV_dict["value_MI_estimate"]
    input_words = QKV_dict["input_words"]

    np.fill_diagonal(query_MI_estimate, 0)
    np.fill_diagonal(key_MI_estimate, 0)
    np.fill_diagonal(value_MI_estimate, 0)

    fig, axs = plt.subplots(1,3)

    img = axs[0].imshow(query_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    img = axs[1].imshow(key_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"MI between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    img = axs[2].imshow(value_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"MI between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # fig.colorbar(img, ax=axs.ravel().tolist())
    fig.suptitle(f"Mutual Information between the rows of Q', K', V' for \nepoch {epoch}, attention layer {attention_layer}, sentence {sentence_id}")

    plt.subplots_adjust(wspace=0.8)
    plt.show(block=True)

def get_JSD_stats(QKV_dict, attention_layers):
    query_JSD_estimate = QKV_dict["query_JSD_estimate"]
    key_JSD_estimate = QKV_dict["key_JSD_estimate"]
    value_JSD_estimate = QKV_dict["value_JSD_estimate"]

    temp_query_JSD_estimate = np.copy(query_JSD_estimate)
    temp_key_JSD_estimate = np.copy(key_JSD_estimate)
    temp_value_JSD_estimate = np.copy(value_JSD_estimate)

    np.fill_diagonal(temp_query_JSD_estimate, 1e9)
    np.fill_diagonal(temp_key_JSD_estimate, 1e9)
    np.fill_diagonal(temp_value_JSD_estimate, 1e9)

    np.fill_diagonal(query_JSD_estimate, 0)
    np.fill_diagonal(key_JSD_estimate, 0)
    np.fill_diagonal(value_JSD_estimate, 0)

    JSD_stats = \
        dict(query_JSD_min = np.min(temp_query_JSD_estimate),
             query_JSD_max = np.max(query_JSD_estimate),
             query_JSD_mean = np.mean(query_JSD_estimate),
             key_JSD_min = np.min(temp_key_JSD_estimate),
             key_JSD_max = np.max(key_JSD_estimate),
             key_JSD_mean = np.mean(key_JSD_estimate),
             value_JSD_min = np.min(temp_value_JSD_estimate),
             value_JSD_max = np.max(value_JSD_estimate),
             value_JSD_mean = np.mean(value_JSD_estimate))

    return JSD_stats

# Main function of this script
def main():
    epoch = 19
    attention_layer_list = [0, 1, 3, 5]
    attention_layer = 3
    sentence_id = 3

    attention_layer_idx = attention_layer_list.index(attention_layer)

    data_dir = Path("data/qkv_prime/model_data")
    if data_dir.exists() == False:
        print(f"Error: The data directory {data_dir} does not exist")
        return  

    data_file = data_dir / f"qkv_prime_epoch_{epoch:02d}_sentence_{sentence_id:02d}.pt"
    if data_file.exists() == False:
        print(f"Error: The data file {data_file} does not exist")
        return

    QKV_list = torch.load(data_file, weights_only=False)
    plot_QKV_prime(QKV_list[attention_layer_idx], epoch, attention_layer, sentence_id)

    # Plot the bar plots of the MI estimates for specific words
    word_list = ["in", "dog", "car", "cat", "van"]
    plot_MI_bars(QKV_list[attention_layer_idx], word_list)
    plot_JSD_bars(QKV_list[attention_layer_idx], word_list)

    epochs_to_analyze = [0, 3, 6, 9, 10, 11, 14, 17, 19]
    JSD_stats_list = list()

    for epoch in epochs_to_analyze:
        data_file = data_dir / f"qkv_prime_epoch_{epoch:02d}_sentence_{sentence_id:02d}.pt"
        if data_file.exists() == False:
            print(f"Error: The data file {data_file} does not exist")
            return

        QKV_list = torch.load(data_file, weights_only=False)
        JSD_stats = get_JSD_stats(QKV_list[attention_layer_idx], attention_layer)
        JSD_stats_list.append(JSD_stats)

    # plot_JSD_estimate(JSD_stats_list, epochs_to_analyze, attention_layer)
    plot_JSD_lineplot(JSD_stats_list, epochs_to_analyze, attention_layer)

# Entry point of the script
if __name__ == '__main__':
    main()