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

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_QKV import get_query_key_value_matrix
from stack_QKV import stack_QKV_matrix
from QKV_entropy import compute_QKV_matrix_entropy
from get_min_max import get_min_max_QKV_matrix



def plot_QKV_matrix_entropy(QKV_entropy_list, N_attention_layers, min_val, max_val, epochs_to_analyze, vector_str, plot_title=None):
    # Plot the entropy values for each dimension of the query/key/value array across epochs
    fig, axs = plt.subplots(2, 3)
    for atten_layer in range(N_attention_layers):
        entropy_array = None

        for epoch in range(len(epochs_to_analyze)):
            epoch_entropy = np.array(QKV_entropy_list[epoch][f'attention_{atten_layer}'][vector_str])
            if entropy_array is None:
                entropy_array = epoch_entropy
            else:
                entropy_array = np.vstack((entropy_array, epoch_entropy))

        a = atten_layer//3
        b = atten_layer%3
        im = axs[a, b].imshow(entropy_array.T, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower', extent=[0, len(epochs_to_analyze)-1, 0, entropy_array.shape[1]-1])
        axs[a, b].set_aspect('auto')
        axs[a, b].set_title(f"Attention Layer {atten_layer}")
        axs[a, b].set_xlabel("Epoch")
        axs[a, b].set_ylabel("Dimension")
        axs[a, b].set_xticks(range(0, len(epochs_to_analyze)), epochs_to_analyze)


    if plot_title:
        fig.suptitle(plot_title)
    else:
        fig.suptitle(f"Entropy of each dimension of the {vector_str} array across epochs")

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show(block=False)



def process_QKV_matrix(analyzer : TransformerAnalyzer):
    print("Computing the entropy of each dimension of the Query, Key and Value arrays ...")

    epochs_to_analyze = np.arange(0, 20, 1)
    # epochs_to_analyze = [0, 4, 9, 14, 19]

    # These will contain the entropy values for the query, key and value arrays
    # for each attention layer for all epochs
    QKV_entropy_list = list()

    # Analyze the probes of the Multihead Attention layers of the encoder for each epoch
    for epoch in epochs_to_analyze:
        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch)

        # Number of input sentences in this epoch
        N_inputs = len(analyzer.encoder_probe._probe_in)

        # This will contain the QKV dictionaries for all the attention layers
        # of all the input sentences of this epoch
        QKV_list = list()

        # Iterate across all the input sentences of this epoch and get the query, key and value arrays.
        # Stack the arrays horizontally after each iteration
        for i in range(N_inputs):
            N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, i)
            # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
            # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

            # Get the query, key, value arrays for all the attention layers of this input sentence
            QKV_dict = get_query_key_value_matrix(analyzer, i, N_src_tokens)
            QKV_list.append(QKV_dict)


        # Concatenate the query, key and value arrays horizontally for all the input sentences of this epoch
        QKV_stacked_dict = stack_QKV_matrix(QKV_list, N_inputs)

        # Compute the entropy and mutual information for each dimension of the query, key and value arrays
        QKV_entropy_dict = compute_QKV_matrix_entropy(QKV_stacked_dict)
        QKV_entropy_list.append(QKV_entropy_dict)

    # Plot the entropy values for each dimension of the query array across epochs
    min_val, max_val = get_min_max_QKV_matrix(QKV_entropy_list, epochs_to_analyze, analyzer.N_attention_layers)
    plot_QKV_matrix_entropy(QKV_entropy_list, analyzer.N_attention_layers, min_val, max_val, epochs_to_analyze, 'query')
    plot_QKV_matrix_entropy(QKV_entropy_list, analyzer.N_attention_layers, min_val, max_val, epochs_to_analyze, 'key')
    plot_QKV_matrix_entropy(QKV_entropy_list, analyzer.N_attention_layers, min_val, max_val, epochs_to_analyze, 'value')
