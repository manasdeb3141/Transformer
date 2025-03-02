# Function to compute the entropy of the Query, Key and Value arrays of the Multihead Attention layer
import sys
sys.path.append('../..')
sys.path.append('../utils')

import torch
import torch.nn as nn
import os
import argparse
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.decomposition import PCA
import platform

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_QKV import get_query_key_value_matrix
from stack_matrices import stack_QKV_matrix
from QKV_joint_pdf import compute_QKV_joint_pdf
from get_min_max import get_min_max_QKV_matrix

from BlahutArimoto import blahut_arimoto_capacity



def plot_QKV_entropy(analyzer, QKV_joint_pdf_list, QKV_dict_list, attention_layer):
    # Plot the entropy values for each dimension

    joint_pdf_dict = QKV_joint_pdf_list[0][f"attention_{attention_layer}"]
    x_entropy_list = joint_pdf_dict["x_entropy_list"]
    q_entropy_list = joint_pdf_dict["q_entropy_list"]
    k_entropy_list = joint_pdf_dict["k_entropy_list"]
    v_entropy_list = joint_pdf_dict["v_entropy_list"]

    QKV_dict = QKV_dict_list[0]
    sentence_tokens = QKV_dict["sentence_tokens"]

    # Get the input sentence words corresponding to the tokens
    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    fig, axs = plt.subplots(2, 2)

    # Plot the entropy across dimensions for the Encoder input
    entropy_array = None
    for x_entropy in x_entropy_list:
        if entropy_array is None:
            entropy_array = x_entropy
        else:
            entropy_array = np.vstack((entropy_array, x_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[0, 0].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[0, 0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0, 0].set_aspect('auto')
    axs[0, 0].set_title(f"Entropy of the Encoder input")
    axs[0, 0].set_ylabel("d_model")

    # Plot the entropy across dimensions for the Q_prime
    entropy_array = None
    for q_entropy in q_entropy_list:
        if entropy_array is None:
            entropy_array = q_entropy
        else:
            entropy_array = np.vstack((entropy_array, q_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[0, 1].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[0, 1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0, 1].set_aspect('auto')
    axs[0, 1].set_title(f"Entropy of Q_prime")
    axs[0, 1].set_ylabel("d_model")


    # Plot the entropy across dimensions for the K_prime
    entropy_array = None
    for k_entropy in k_entropy_list:
        if entropy_array is None:
            entropy_array = k_entropy
        else:
            entropy_array = np.vstack((entropy_array, k_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[1, 0].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[1, 0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1, 0].set_aspect('auto')
    axs[1, 0].set_title(f"Entropy of K_prime")
    axs[1, 0].set_ylabel("d_model")


    # Plot the entropy across dimensions for the V_prime
    entropy_array = None
    for v_entropy in v_entropy_list:
        if entropy_array is None:
            entropy_array = v_entropy
        else:
            entropy_array = np.vstack((entropy_array, v_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[1, 1].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[1, 1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1, 1].set_aspect('auto')
    axs[1, 1].set_title(f"Entropy of V_prime")
    axs[1, 1].set_ylabel("d_model")

    fig.suptitle(f"Entropy of the Encoder input and Q_prime, K_prime and V_prime")

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Get the current figure manager to maximize the plot window
    fig_manager = plt.get_current_fig_manager()
    if platform.system() == 'Linux':
        fig_manager.window.showMaximized()
    else:
        fig_manager.window.state('zoomed')

    plt.savefig(f"data/QKV_prime_entropy.png")
    plt.show(block=True)


def plot_QKV_joint_entropy(analyzer, QKV_joint_pdf_list, QKV_dict_list, attention_layer):
    # Plot the entropy values for each dimension

    joint_pdf_dict = QKV_joint_pdf_list[0][f"attention_{attention_layer}"]
    x_entropy_list = joint_pdf_dict["x_entropy_list"]
    xq_entropy_list = joint_pdf_dict["xq_entropy_list"]
    xk_entropy_list = joint_pdf_dict["xk_entropy_list"]
    xv_entropy_list = joint_pdf_dict["xv_entropy_list"]

    QKV_dict = QKV_dict_list[0]
    sentence_tokens = QKV_dict["sentence_tokens"]

    # Get the input sentence words corresponding to the tokens
    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    fig, axs = plt.subplots(2, 2)

    # Plot the entropy across dimensions for the Encoder input
    entropy_array = None
    for x_entropy in x_entropy_list:
        if entropy_array is None:
            entropy_array = x_entropy
        else:
            entropy_array = np.vstack((entropy_array, x_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[0, 0].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[0, 0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0, 0].set_aspect('auto')
    axs[0, 0].set_title(f"Entropy of the Encoder input")
    axs[0, 0].set_ylabel("d_model")

    # Plot the entropy across dimensions for the Q_prime
    entropy_array = None
    for xq_entropy in xq_entropy_list:
        if entropy_array is None:
            entropy_array = xq_entropy
        else:
            entropy_array = np.vstack((entropy_array, xq_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[0, 1].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[0, 1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0, 1].set_aspect('auto')
    axs[0, 1].set_title(f"Joint entropy of the Encoder input and Q_prime")
    axs[0, 1].set_ylabel("d_model")


    # Plot the entropy across dimensions for the K_prime
    entropy_array = None
    for xk_entropy in xk_entropy_list:
        if entropy_array is None:
            entropy_array = xk_entropy
        else:
            entropy_array = np.vstack((entropy_array, xk_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[1, 0].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[1, 0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1, 0].set_aspect('auto')
    axs[1, 0].set_title(f"Joint entropy of the Encoder input and K_prime")
    axs[1, 0].set_ylabel("d_model")


    # Plot the entropy across dimensions for the V_prime
    entropy_array = None
    for xv_entropy in xv_entropy_list:
        if entropy_array is None:
            entropy_array = xv_entropy
        else:
            entropy_array = np.vstack((entropy_array, xv_entropy))

    # im = axs[0, 0].imshow(entropy_array, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower')
    im = axs[1, 1].imshow(entropy_array.T, cmap=plt.cm.jet, origin='lower')
    axs[1, 1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1, 1].set_aspect('auto')
    axs[1, 1].set_title(f"Joint entropy of the Encoder input and V_prime")
    axs[1, 1].set_ylabel("d_model")

    fig.suptitle(f"Joint Entropy of the Encoder input and Q_prime, K_prime and V_prime")

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Get the current figure manager to maximize the plot window
    fig_manager = plt.get_current_fig_manager()
    if platform.system() == 'Linux':
        fig_manager.window.showMaximized()
    else:
        fig_manager.window.state('zoomed')

    plt.savefig(f"data/QKV_prime_joint_entropy.png")
    plt.show(block=True)


def compute_QKV_capacity(QKV_joint_pdf_list, epochs_to_analyze, attention_layer):
    # This will contain the capacity values for the W_q, W_k, W_v matrices
    capacity_dict = dict()

    joint_pdf_dict = QKV_joint_pdf_list[0][f"attention_{attention_layer}"]
    Wq_joint_pdf = joint_pdf_dict["Wq_joint_pdf"]
    P_XY = Wq_joint_pdf["P_XY"]
    P_X = Wq_joint_pdf["P_X"]
    P_Y = Wq_joint_pdf["P_Y"]
    P_Y_given_X = P_XY / P_X
    row_sum = P_Y_given_X.sum(axis=1, keepdims=True)
    P_Y_given_X = P_Y_given_X / row_sum
    C_Wq, _ = blahut_arimoto_capacity(P_Y_given_X)

    Wk_joint_pdf = joint_pdf_dict["Wk_joint_pdf"]
    P_XY = Wq_joint_pdf["P_XY"]
    P_X = Wq_joint_pdf["P_X"]
    P_Y = Wq_joint_pdf["P_Y"]
    P_Y_given_X = P_XY / P_X
    row_sum = P_Y_given_X.sum(axis=1, keepdims=True)
    P_Y_given_X = P_Y_given_X / row_sum
    C_Wk, _ = blahut_arimoto_capacity(P_Y_given_X)

    Wv_joint_pdf = joint_pdf_dict["Wv_joint_pdf"]
    P_XY = Wq_joint_pdf["P_XY"]
    P_X = Wq_joint_pdf["P_X"]
    P_Y = Wq_joint_pdf["P_Y"]
    P_Y_given_X = P_XY / P_X
    row_sum = P_Y_given_X.sum(axis=1, keepdims=True)
    P_Y_given_X = P_Y_given_X / row_sum
    C_Wv, _ = blahut_arimoto_capacity(P_Y_given_X)

    capacity_dict["C_Wq"] = C_Wq
    capacity_dict["C_Wk"] = C_Wk    
    capacity_dict["C_Wv"] = C_Wv

    return capacity_dict


def process_QKV_capacity(analyzer : TransformerAnalyzer):
    print("Computing the capacity and joint PDF of the input and output of the W_q, W_k, W_v matrices ...")
    QKV_joint_pdf_filename = 'data/QKV_joint_pdf_list.pt'

    epochs_to_analyze = np.arange(0, 20, 1)
    # epochs_to_analyze = [0, 4, 9, 14, 19]
    epochs_to_analyze = [19]
    attention_layer = 1

    QKV_joint_pdf_list = None
    QKV_joint_pdf_file = Path(QKV_joint_pdf_filename)
    if QKV_joint_pdf_file.is_file():
        print("QKV joint PDF list file found. Loading it ...")
        QKV_combined_dict = torch.load(QKV_joint_pdf_file, weights_only=False)
        QKV_joint_pdf_list = QKV_combined_dict["QKV_joint_pdf_list"]
        QKV_list = QKV_combined_dict["QKV_list"]

    if QKV_joint_pdf_list is None:
        # This will contain the entropy values for the query, key and value arrays
        # for each attention layer for all epochs
        QKV_joint_pdf_list = list()

        # Analyze the probes of the Multihead Attention layers of the encoder for each epoch
        for epoch in epochs_to_analyze:
            # For this epoch, load all the encoder layer probe files from disk
            analyzer.load_encoder_probes(epoch)

            # Number of input sentences in this epoch
            N_inputs = len(analyzer.encoder_probe._probe_in)
            # Manas
            N_inputs = 1

            # This will contain the QKV dictionaries for all the attention layers
            # of all the input sentences of this epoch
            QKV_list = list()

            # Iterate across all the input sentences of this epoch and get the query, key and value arrays.
            # Stack the arrays horizontally after each iteration
            for i in range(N_inputs):
                # Manas
                i = 3 
                N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, i)
                # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
                # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

                # Get the query, key, value arrays for all the attention layers of this input sentence
                QKV_dict = get_query_key_value_matrix(analyzer, i, N_src_tokens)
                QKV_dict["sentence_tokens"] = src_sentence_tokens
                QKV_list.append(QKV_dict)

            # Concatenate the query, key and value arrays horizontally for all the input sentences of this epoch
            QKV_stacked_dict = stack_QKV_matrix(QKV_list, N_inputs)

            # Compute the entropy and mutual information for each dimension of the query, key and value arrays
            print(f"Computing the entropy for epoch {epoch} ...")
            QKV_joint_pdf_dict = compute_QKV_joint_pdf(QKV_stacked_dict, attention_layer, N_src_tokens)
            QKV_joint_pdf_list.append(QKV_joint_pdf_dict)

        # Save the entropy values for each dimension of the Q,K,V arrays across epochs
        QKV_combined_dict = dict(QKV_joint_pdf_list=QKV_joint_pdf_list, QKV_list=QKV_list)
        torch.save(QKV_combined_dict, QKV_joint_pdf_filename)
        print(f"QKV joint PDF list saved to file {QKV_joint_pdf_filename}.")
        
    capacity_dict = compute_QKV_capacity(QKV_joint_pdf_list, epochs_to_analyze, attention_layer)
    print(f"W_q capacity = {capacity_dict["C_Wq"]}, W_k capacity = {capacity_dict["C_Wk"]}, W_v capacity = {capacity_dict["C_Wv"]}")

    # Plot the entropy values for each dimension of the q_prime, k_prime, v_prime arrays
    plot_QKV_entropy(analyzer, QKV_joint_pdf_list, QKV_list, attention_layer)
    plot_QKV_joint_entropy(analyzer, QKV_joint_pdf_list, QKV_list, attention_layer)