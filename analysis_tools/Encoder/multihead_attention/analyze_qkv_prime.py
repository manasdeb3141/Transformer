
import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

def get_top_N_values(input_vals, N):
    # Get the indices of the N largest elements
    indices = np.argpartition(input_vals, N)[-N:]

    # Return the N largest elements and their indices
    return input_vals[indices], indices

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


def plot_MI_bars(QKV_dict, word_list):
    query_MI_estimate = QKV_dict["query_MI_estimate"]
    key_MI_estimate = QKV_dict["key_MI_estimate"]
    value_MI_estimate = QKV_dict["value_MI_estimate"]
    input_words = QKV_dict["input_words"]

    np.fill_diagonal(query_MI_estimate, 0)
    np.fill_diagonal(key_MI_estimate, 0)
    np.fill_diagonal(value_MI_estimate, 0)

    for word in word_list:
        word_index = input_words.index(word)
        query_MI = query_MI_estimate[word_index]
        key_MI = key_MI_estimate[word_index]
        value_MI = value_MI_estimate[word_index]

        fig, axs = plt.subplots(3,1)
        x_vals = np.arange(0, len(input_words))

        # Plot Q'
        top_vals, N_top_indices = get_top_N_values(query_MI, 3)
        top_indices = np.argwhere(query_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        barplot_Q = axs[0].bar(x_vals, query_MI, color=bar_colors)
        axs[0].set_title(f"Q'")
        axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)

        # Plot K'
        top_vals, N_top_indices = get_top_N_values(key_MI, 3)
        top_indices = np.argwhere(key_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        barplot_K = axs[1].bar(x_vals, key_MI, color=bar_colors)
        axs[1].set_title(f"K'")
        axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)

        # Plot V'
        top_vals, N_top_indices = get_top_N_values(value_MI, 3)
        top_indices = np.argwhere(value_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        barplot_V = axs[2].bar(x_vals, value_MI, color=bar_colors)
        axs[2].set_title(f"V'")
        axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)

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
    axs[0].set_aspect('auto')
    axs[0].set_title(f"Mutual Information between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(key_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('auto')
    axs[1].set_title(f"Mutual Information between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(value_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('auto')
    axs[2].set_title(f"Mutual Information between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=axs[2])

    fig.suptitle(f"Mutual Information between the rows of Q', K', V' for \nepoch {epoch}, attention layer {attention_layer}, sentence {sentence_id}")

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
    attention_layer_idx = 2
    attention_layer = 0
    sentence_id = 3

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
    word_list = ["dog", "car", "cat", "van"]
    plot_MI_bars(QKV_list[attention_layer_idx], word_list)

    epochs_to_analyze = [0, 9, 19]
    attention_layers = [0, 1, 5]
    JSD_stats_list = list()

    for epoch in epochs_to_analyze:
        data_file = data_dir / f"qkv_prime_epoch_{epoch:02d}_sentence_{sentence_id:02d}.pt"
        if data_file.exists() == False:
            print(f"Error: The data file {data_file} does not exist")
            return

        QKV_list = torch.load(data_file, weights_only=False)
        JSD_stats = get_JSD_stats(QKV_list[attention_layer_idx], attention_layer)
        JSD_stats_list.append(JSD_stats)

    plot_JSD_estimate(JSD_stats_list, epochs_to_analyze, attention_layer)

# Entry point of the script
if __name__ == '__main__':
    main()