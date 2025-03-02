

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
from get_QKV import get_cross_atten_QKV
from get_sentence_tokens import get_sentence_tokens

def plot_cross_atten_QKV_prime_MI(QKV_MI_dict, epoch, decoder_token_id, attention_layer):
    input_words = QKV_MI_dict["input_words"]
    QQ_MI_estimate = QKV_MI_dict["QQ_MI_estimate"]
    KK_MI_estimate = QKV_MI_dict["KK_MI_estimate"]
    VV_MI_estimate = QKV_MI_dict["VV_MI_estimate"]
    QK_MI_estimate = QKV_MI_dict["QK_MI_estimate"]
    QV_MI_estimate = QKV_MI_dict["QV_MI_estimate"]


    # 1st plot: Mutual Information between the rows of Q', K', V'
    fig, axs = plt.subplots(1,3)

    img = axs[0].imshow(QQ_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('auto')
    axs[0].set_title(f"Mutual Information between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(KK_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('auto')
    axs[1].set_title(f"Mutual Information between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(VV_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('auto')
    axs[2].set_title(f"Mutual Information between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[2])

    fig.suptitle(f"Mutual Information between the decoder inputs for epoch {epoch}, token #{decoder_token_id}, attention layer {attention_layer}")

    plt.show(block=True)


    # 2nd plot : Q'Q', Q'K', Q'V'
    fig, axs = plt.subplots(1,3)

    img = axs[0].imshow(QQ_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('auto')
    axs[0].set_title(f"Mutual Information between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(QK_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('auto')
    axs[1].set_title(f"Mutual Information between the rows of Q' and K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(QV_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('auto')
    axs[2].set_title(f"Mutual Information between the rows of Q' and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[2])

    fig.suptitle(f"Mutual Information between the decoder inputs for epoch {epoch}, token #{decoder_token_id}, attention layer {attention_layer}")

    plt.show(block=True)

def plot_cross_atten_QKV_MI(QKV_MI_dict, epoch, decoder_token_id, attention_layer):
    input_words = QKV_MI_dict["input_words"]
    QQ_MI_estimate = QKV_MI_dict["QQ_MI_estimate"]
    QK_MI_estimate = QKV_MI_dict["QK_MI_estimate"]
    QV_MI_estimate = QKV_MI_dict["QV_MI_estimate"]

    fig, axs = plt.subplots(1,3)

    img = axs[0].imshow(QQ_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('auto')
    axs[0].set_title(f"Mutual Information between the rows of Q")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(QK_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('auto')
    axs[1].set_title(f"Mutual Information between the rows of Q and K")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(QV_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('auto')
    axs[2].set_title(f"Mutual Information between the rows of Q and V")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[2])

    fig.suptitle(f"Mutual Information between the decoder inputs for epoch {epoch}, token #{decoder_token_id}, attention layer {attention_layer}")

    plt.show(block=True)

 
def process_cross_atten_QKV_prime(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens):
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

    for i, j in tqdm(ij_pos):
        X_row = Q_prime[i]
        Y_row = Q_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        QQ_MI_estimate[i, j] = MI

    print("\nComputing the mutual information between the rows of K ...")
    # Initialize the mutual information matrix
    KK_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = K_prime[i]
        Y_row = K_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        KK_MI_estimate[i, j] = MI

    print("\nComputing the mutual information between the rows of V ...")
    # Initialize the mutual information matrix
    VV_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = V_prime[i]
        Y_row = V_prime[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        VV_MI_estimate[i, j] = MI

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

    # Zero out the diagonal elements so that the non-diagnal elements are visible
    np.fill_diagonal(QQ_MI_estimate, 0)
    np.fill_diagonal(KK_MI_estimate, 0)
    np.fill_diagonal(VV_MI_estimate, 0)
    np.fill_diagonal(QK_MI_estimate, 0)
    np.fill_diagonal(QV_MI_estimate, 0)

    # Get the target words from the tokens
    input_words = list()
    for token in tgt_sentence_tokens:
            input_words.append(analyzer.get_tgt_word_from_token(token))

    return dict(input_words=input_words, QQ_MI_estimate=QQ_MI_estimate, KK_MI_estimate=KK_MI_estimate, VV_MI_estimate=VV_MI_estimate, QK_MI_estimate=QK_MI_estimate, QV_MI_estimate=QV_MI_estimate)


def process_cross_atten_QKV_inputs(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens):
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

    # Zero out the diagonal elements so that the non-diagnal elements are visible
    np.fill_diagonal(QQ_MI_estimate, 0)
    np.fill_diagonal(QK_MI_estimate, 0)
    np.fill_diagonal(QV_MI_estimate, 0)

    # Get the target words from the tokens
    input_words = list()
    for token in tgt_sentence_tokens:
            input_words.append(analyzer.get_tgt_word_from_token(token))

    return dict(input_words=input_words, QQ_MI_estimate=QQ_MI_estimate, QK_MI_estimate=QK_MI_estimate, QV_MI_estimate=QV_MI_estimate)


def main():
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

    QKV_dict = get_cross_atten_QKV(analyzer, sentence_id, decoder_token_id, N_src_tokens)

    # QKV_MI_dict = process_cross_atten_QKV_inputs(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens)
    # plot_cross_atten_QKV_prime_MI(QKV_MI_dict, epoch, decoder_token_id, attention_layer)

    QKV_prime_MI_dict = process_cross_atten_QKV_prime(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens)
    plot_cross_atten_QKV_prime_MI(QKV_prime_MI_dict, epoch, decoder_token_id, attention_layer)


# Entry point of the program
if __name__ == '__main__':
    main()