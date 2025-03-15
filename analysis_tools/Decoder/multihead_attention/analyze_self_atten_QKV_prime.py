

import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import torch
import torch.nn as nn
import torchinfo
import sys
import numpy as np
from pathlib import Path
from termcolor import cprint, colored
import matplotlib.pyplot as plt
from tqdm import tqdm

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

# Class implemented by this application
from Transformer import Transformer
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer
from mutual_info_estimator import MutualInfoEstimator

# Functions implemented by this application
from get_QKV import get_self_atten_QKV
from get_sentence_tokens import get_sentence_tokens

def get_top_N_values(input_vals, N):
    # Get the indices of the N largest elements
    indices = np.argpartition(input_vals, N)[-N:]

    # Return the N largest elements and their indices
    return input_vals[indices], indices

def plot_MI_bars_QK_prime(QKV_dict, word_list):
    QK_MI_estimate = QKV_dict["QK_MI_estimate"]
    input_words = QKV_dict["input_words"]

    for word in word_list:
        word_index = input_words.index(word)
        QK_MI = QK_MI_estimate[word_index]

        fig, ax = plt.subplots()
        x_vals = np.arange(0, len(input_words))

        # Plot MI between the rows of Q' and K' prime
        top_vals, N_top_indices = get_top_N_values(QK_MI, 3)
        top_indices = np.argwhere(QK_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_Q = ax.bar(x_vals, QK_MI, color=bar_colors)
        ax.set_title(f"MI between the row of Q' and all the rows of K' for the word '{word}'")
        ax.set_xticks(range(0, len(input_words)), input_words, rotation=90)
        ax.set_ylabel("Mutual Information")
        for bar in barplot_Q:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        fig.suptitle(f"Mutual Information between the word '{word}' and other words")
        plt.show(block=True)


def plot_MI_bars_QKV_prime(QKV_dict, word_list):
    query_MI_estimate = QKV_dict["QQ_MI_estimate"]
    key_MI_estimate = QKV_dict["KK_MI_estimate"]
    value_MI_estimate = QKV_dict["VV_MI_estimate"]
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
        bar_colors = ['red' if i in top_indices else 'gray' for i in range(len(input_words))]
        barplot_Q = axs[0].bar(x_vals, query_MI, color=bar_colors)
        axs[0].set_title(f"Q'")
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
        axs[1].set_title(f"K'")
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
        axs[2].set_title(f"V'")
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

def plot_JSD_estimate(QKV_MI_dict, epoch, decoder_token_id, attention_layer):
    input_words = QKV_MI_dict["input_words"]
    QQ_JSD_estimate = QKV_MI_dict["QQ_JSD_estimate"]
    KK_JSD_estimate = QKV_MI_dict["KK_JSD_estimate"]
    VV_JSD_estimate = QKV_MI_dict["VV_JSD_estimate"]

    fig, axs = plt.subplots(1,3)

    max_val = np.max([np.max(QQ_JSD_estimate), np.max(KK_JSD_estimate), np.max(VV_JSD_estimate)])

    # img = axs[0].imshow(QQ_JSD_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    img = axs[0].imshow(QQ_JSD_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_JSD_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"JSD between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[0])

    # img = axs[1].imshow(KK_JSD_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    img = axs[1].imshow(KK_JSD_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_JSD_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"JSD between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[1])

    # img = axs[2].imshow(VV_JSD_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    img = axs[2].imshow(VV_JSD_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_JSD_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"JSD between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[2])

    # fig.colorbar(img, ax=axs.ravel().tolist())
    fig.suptitle(f"Jensen-Shannon divergence inputs for\nepoch {epoch}, token #{decoder_token_id}, attention layer {attention_layer}")
    plt.show(block=True)

def plot_self_atten_QKV_prime_MI(QKV_MI_dict, epoch, decoder_token_id, attention_layer):
    input_words = QKV_MI_dict["input_words"]
    QQ_MI_estimate = QKV_MI_dict["QQ_MI_estimate"]
    KK_MI_estimate = QKV_MI_dict["KK_MI_estimate"]
    VV_MI_estimate = QKV_MI_dict["VV_MI_estimate"]
    QK_MI_estimate = QKV_MI_dict["QK_MI_estimate"]
    QV_MI_estimate = QKV_MI_dict["QV_MI_estimate"]

    # Zero out the diagonal elements so that the non-diagnal elements are visible
    np.fill_diagonal(QQ_MI_estimate, 0)
    np.fill_diagonal(KK_MI_estimate, 0)
    np.fill_diagonal(VV_MI_estimate, 0)
    # np.fill_diagonal(QK_MI_estimate, 0)
    # np.fill_diagonal(QV_MI_estimate, 0)

    # 1st plot: Mutual Information between the rows of Q', K', V'
    fig, axs = plt.subplots(1,3)

    max_val = np.max([np.max(QQ_MI_estimate), np.max(KK_MI_estimate), np.max(VV_MI_estimate)])

    img = axs[0].imshow(QQ_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(KK_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"MI between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(VV_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"MI between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[2])

    fig.colorbar(img, ax=axs.ravel().tolist())
    fig.suptitle(f"Mutual Information between the decoder inputs for epoch {epoch}, token #{decoder_token_id}, attention layer {attention_layer}")

    plt.show(block=True)

    all_plots = False

    if all_plots:
        # 2nd plot : Q'Q', Q'K', Q'V'
        fig, axs = plt.subplots(1,3)

        img = axs[0].imshow(QQ_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
        # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
        axs[0].set_aspect('equal')
        axs[0].set_title(f"MI between the rows of Q'")
        axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
        # fig.colorbar(img, ax=axs[0])

        img = axs[1].imshow(QK_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
        # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
        axs[1].set_aspect('equal')
        axs[1].set_title(f"MI between the rows of Q' and K'")
        axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
        # fig.colorbar(img, ax=axs[1])

        img = axs[2].imshow(QV_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
        # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
        axs[2].set_aspect('equal')
        axs[2].set_title(f"MI between the rows of Q' and V'")
        axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
        # fig.colorbar(img, ax=axs[2])
        fig.colorbar(img, ax=axs.ravel().tolist())
    else:
        fig, ax = plt.subplots()

        #img = axs[1].imshow(QK_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
        img = ax.imshow(QK_MI_estimate.T, cmap=plt.cm.Wistia)
        # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
        ax.set_aspect('equal')
        ax.set_title(f"MI between the rows of Q' and K'")
        ax.set_xticks(range(0, len(input_words)), input_words, rotation=90)
        ax.set_yticks(range(0, len(input_words)), input_words, rotation=0)
        fig.colorbar(img, ax=ax)

    fig.suptitle(f"Mutual Information between the decoder inputs for epoch {epoch}, token #{decoder_token_id}, attention layer {attention_layer}")

    plt.show(block=True)

 
def process_self_atten_QKV_prime(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens):
    QKV_atten_dict = QKV_dict[f"attention_{attention_layer}"]
    Q_prime = QKV_atten_dict["Q_prime"]
    K_prime = QKV_atten_dict["K_prime"]
    V_prime = QKV_atten_dict["V_prime"]

    # Number of rows of the Query input matrix of the decoder
    N_rows = Q_prime.shape[0] 

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between the rows of Q ...")
    # Initialize the mutual information matrix
    QQ_MI_estimate = np.zeros((N_rows, N_rows))
    QQ_JSD_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = Q_prime[i]
        Y_row = Q_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        prob_data, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        P_X = prob_data["P_X"]
        P_Y = prob_data["P_Y"]
        JSD = MI_estimator.JS_divergence(P_X, P_Y)
        MI = MI_data["MI"]
        QQ_MI_estimate[i, j] = MI
        QQ_JSD_estimate[i, j] = JSD

    print("\nComputing the mutual information between the rows of K ...")
    # Initialize the mutual information matrix
    KK_MI_estimate = np.zeros((N_rows, N_rows))
    KK_JSD_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = K_prime[i]
        Y_row = K_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        prob_data, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        P_X = prob_data["P_X"]
        P_Y = prob_data["P_Y"]
        JSD = MI_estimator.JS_divergence(P_X, P_Y)
        KK_MI_estimate[i, j] = MI
        KK_JSD_estimate[i, j] = JSD

    print("\nComputing the mutual information between the rows of V ...")
    # Initialize the mutual information matrix
    VV_MI_estimate = np.zeros((N_rows, N_rows))
    VV_JSD_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = V_prime[i]
        Y_row = V_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        prob_data, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        P_X = prob_data["P_X"]
        P_Y = prob_data["P_Y"]
        JSD = MI_estimator.JS_divergence(P_X, P_Y)
        MI = MI_data["MI"]
        VV_MI_estimate[i, j] = MI
        VV_JSD_estimate[i, j] = JSD

    print("\nComputing the mutual information between the rows of Q and K ...")
    # Initialize the mutual information matrix
    QK_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = Q_prime[i]
        Y_row = K_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        QK_MI_estimate[i, j] = MI


    print("\nComputing the mutual information between the rows of Q and V ...")
    # Initialize the mutual information matrix
    QV_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = Q_prime[i]
        Y_row = V_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        QV_MI_estimate[i, j] = MI

    # Get the target words from the tokens
    input_words = list()
    for token in tgt_sentence_tokens:
            input_words.append(analyzer.get_tgt_word_from_token(token))

    return dict(input_words=input_words, QQ_MI_estimate=QQ_MI_estimate, QQ_JSD_estimate=QQ_JSD_estimate, KK_MI_estimate=KK_MI_estimate, KK_JSD_estimate=KK_JSD_estimate, VV_MI_estimate=VV_MI_estimate, VV_JSD_estimate=VV_JSD_estimate, QK_MI_estimate=QK_MI_estimate, QV_MI_estimate=QV_MI_estimate)


def process_self_atten_QKV_inputs(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens):
    QKV_atten_dict = QKV_dict[f"attention_{attention_layer}"]
    Q = QKV_atten_dict["Q"]
    K = QKV_atten_dict["K"]
    V = QKV_atten_dict["V"]

    # Number of rows of the Query input matrix of the decoder
    N_rows = Q.shape[0] 

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between the rows of Q ...")
    # Initialize the mutual information matrix
    QQ_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = Q[i]
        Y_row = Q[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        QQ_MI_estimate[i, j] = MI


    print("\nComputing the mutual information between the rows of Q and K ...")
    # Initialize the mutual information matrix
    QK_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = Q[i]
        Y_row = K[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        QK_MI_estimate[i, j] = MI


    print("\nComputing the mutual information between the rows of Q and V ...")
    # Initialize the mutual information matrix
    QV_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = Q[i]
        Y_row = V[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        QV_MI_estimate[i, j] = MI

    # Get the target words from the tokens
    input_words = list()
    for token in tgt_sentence_tokens:
            input_words.append(analyzer.get_tgt_word_from_token(token))

    return dict(input_words=input_words, QQ_MI_estimate=QQ_MI_estimate, QK_MI_estimate=QK_MI_estimate, QV_MI_estimate=QV_MI_estimate)


def main():
    print("Computing the self-attention QKV inputs ...")

    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../../model_data/opus_books_en_fr/tokens"
    model_config["analyze_dir"] = "../../../model_data/opus_books_en_fr/probes_8"
    model_config["d_model"] = 512

    # model_config["tokenizer_dir"] = "../../../model_data_d32/opus_books_en_fr/tokens"
    # model_config["analyze_dir"] = "../../../model_data_d32/opus_books_en_fr/probes_8"
    # model_config["d_model"] = 32

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Analysis parameters
    epoch = 19
    attention_layer = 0
    sentence_id = 3
    decoder_token_id = 10

    save_file = Path(f"data/self_QKV/QKV_self_epoch_{epoch}_layer_{attention_layer}_sentence_{sentence_id}_token_{decoder_token_id}.pt")

    if save_file.exists():
        print(f"QKV self-attention data file {str(save_file)} found. Loading it ...")
        self_QKV_prime_save_dict = torch.load(save_file, weights_only=False)
        QKV_MI_dict = self_QKV_prime_save_dict["QKV_MI_dict"]
        self_QKV_prime_MI_dict = self_QKV_prime_save_dict["self_QKV_prime_MI_dict"]
    else:
        # Instantiate the Transformer's analyzer object.
        # Load the tokenizer and instantiate the ProbeManager objects
        analyzer = TransformerAnalyzer(model_config, probe_config)
        analyzer.run()

        # For this epoch, load all the decoder probe files from disk
        analyzer.load_decoder_probes(epoch)

        # Number of input sentences in this epoch
        N_inputs = len(analyzer.decoder_probe._probe_in)

        # Get the tokens input to the decoder
        N_src_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id, decoder_token_id)

        QKV_dict = get_self_atten_QKV(analyzer, sentence_id, decoder_token_id)

        QKV_MI_dict = process_self_atten_QKV_inputs(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens)
        self_QKV_prime_MI_dict = process_self_atten_QKV_prime(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens)

        self_QKV_prime_save_dict = dict(QKV_MI_dict=QKV_MI_dict, self_QKV_prime_MI_dict=self_QKV_prime_MI_dict)

        # Save the file
        torch.save(self_QKV_prime_save_dict, save_file)

    word_list = ["chien", "dans", "wagon"]
    plot_self_atten_QKV_prime_MI(self_QKV_prime_MI_dict, epoch, decoder_token_id, attention_layer)
    # plot_MI_bars_QKV_prime(self_QKV_prime_MI_dict, word_list) 
    plot_MI_bars_QK_prime(self_QKV_prime_MI_dict, word_list)

    plot_JSD_estimate(self_QKV_prime_MI_dict, epoch, decoder_token_id, attention_layer)

# Entry point of the program
if __name__ == '__main__':
    main()