# Function to compute the entropy and mutual information of the Query, Key and Value heads of the Multihead Attention layer
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
from get_QKV import get_query_key_value_head
from stack_matrices import stack_QKV_matrix
from QKV_head_entropy import compute_QKV_head_entropy
from QKV_head_mi import compute_QKV_head_mi
from get_min_max import get_min_max_QKV_head

def plot_QKV_head_entropy(QKV_entropy_list, N_attention_layers, min_val, max_val, epochs_to_analyze, vector_str, plot_title=None):
    # Plot the entropy values for each dimension of the query/key/value array across epochs
    for atten_layer in range(N_attention_layers):
        attn_entropy_list = np.array(QKV_entropy_list[0][f'attention_{atten_layer}'][vector_str])
        N_heads = len(attn_entropy_list)
        fig, axs = plt.subplots(2, 4)

        for head in range(N_heads):
            head_entropy_array = None

            for epoch in range(len(epochs_to_analyze)):
                epoch_head_entropy = np.array(QKV_entropy_list[epoch][f'attention_{atten_layer}'][vector_str][head])
                if head_entropy_array is None:
                    head_entropy_array = epoch_head_entropy
                else:
                    head_entropy_array = np.vstack((head_entropy_array, epoch_head_entropy))

            a = head//4
            b = head%4
            im = axs[a, b].imshow(head_entropy_array.T, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower', extent=[0, len(epochs_to_analyze)-1, 0, head_entropy_array.shape[1]-1])
            axs[a, b].set_aspect('auto')
            axs[a, b].set_title(f"Head {head}")
            axs[a, b].set_xlabel("Epoch")
            if b == 0:
                axs[a, b].set_ylabel("Dimension")
            axs[a, b].set_xticks([0, 4, 9, 14, 19])

        if plot_title:
            fig.suptitle(plot_title)
        else:
            fig.suptitle(f"Entropy of the {vector_str} head for attention layer {atten_layer}")

        plt.subplots_adjust(hspace=0.8, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        # plt.tight_layout()

        plt.show(block=False)

def plot_QKV_head_mi(self, QKV_mi_list, N_attention_layers, min_val, max_val, N_src_tokens, epochs_to_analyze, selection, MI_str) -> None :
    # Plot the mutual information values between the query/key/value array across epochs
    for atten_layer in range(N_attention_layers):
        attn_mi_list = np.array(QKV_mi_list[0][f'attention_{atten_layer}'][selection])
        N_heads = len(attn_mi_list)
        fig, axs = plt.subplots(2, 4)

        for head in range(N_heads):
            head_mi_array = None

            for epoch in range(len(epochs_to_analyze)):
                epoch_head_mi = np.array(QKV_mi_list[epoch][f'attention_{atten_layer}'][selection][head])
                if head_mi_array is None:
                    head_mi_array = epoch_head_mi
                else:
                    head_mi_array = np.vstack((head_mi_array, epoch_head_mi))

            a = head//4
            b = head%4
            im = axs[a, b].imshow(head_mi_array.T, cmap=plt.cm.jet, vmin=min_val, vmax=max_val, origin='lower', extent=[0, len(epochs_to_analyze)-1, 0, head_mi_array.shape[1]-1])
            axs[a, b].set_aspect('auto')
            axs[a, b].set_title(f"Head {head}")
            axs[a, b].set_xlabel("Epoch")
            if b == 0:
                axs[a, b].set_ylabel("Token Position")
            axs[a, b].set_xticks([0, 4, 9, 14, 19])
            axs[a, b].set_yticks(np.arange(0, N_src_tokens, 2))

        fig.suptitle(f"Mutual Information of the {MI_str} head for attention layer {atten_layer}")
        plt.subplots_adjust(hspace=0.8, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show(block=False)

def process_QKV_heads(analyzer : TransformerAnalyzer):
    print("Computing the entropy and mutual information of each dimension of the Query, Key and Value heads ...")
    epochs_to_analyze = np.arange(0, 20, 1)
    # epochs_to_analyze = [0, 4, 9, 14, 19]

    # These will contain entropy and mutual information values for the query, key and value head arrays
    # for each attention layer for all epochs. Each element in these lists are a list of values for each head
    QKV_entropy_list = list()
    QKV_mi_list = list()

    # Input sentence ID
    sentence_id = 7

    # Analyze the probes of the Multihead Attention layers of the encoder for each epoch
    for epoch in epochs_to_analyze:
        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch)

        # Get the number of tokens of the sentence
        N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)
        # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
        # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

        # Get the query, key, value arrays for all the attention layers of this input sentence
        # N_src_tokens=350
        QKV_dict = get_query_key_value_head(analyzer, sentence_id, N_src_tokens)

        # Compute the entropy and mutual information for each dimension of the query, key and value arrays
        QKV_entropy_dict = compute_QKV_head_entropy(QKV_dict, analyzer.N_attention_layers)
        QKV_mi_dict = compute_QKV_head_mi(QKV_dict, analyzer.N_attention_layers)

        QKV_entropy_list.append(QKV_entropy_dict)
        QKV_mi_list.append(QKV_mi_dict)

    # Plot the entropy values for each dimension of the query array across epochs
    min_val, max_val = get_min_max_QKV_head(QKV_entropy_list, epochs_to_analyze, analyzer.N_attention_layers, True)
    plot_QKV_head_entropy(QKV_entropy_list, analyzer.N_attention_layers, min_val, max_val, epochs_to_analyze, "query")
    plot_QKV_head_entropy(QKV_entropy_list, analyzer.N_attention_layers, min_val, max_val, epochs_to_analyze, "key")
    plot_QKV_head_entropy(QKV_entropy_list, analyzer.N_attention_layers, min_val, max_val, epochs_to_analyze, "value")

    min_val, max_val = get_min_max_QKV_head(QKV_mi_list, epochs_to_analyze, analyzer.N_attention_layers, False)
    plot_QKV_head_mi(QKV_mi_list, analyzer.N_attention_layers, min_val, max_val, N_src_tokens, epochs_to_analyze, "qk_mi", "Query-Key")
    plot_QKV_head_mi(QKV_mi_list, analyzer.N_attention_layers, min_val, max_val, N_src_tokens, epochs_to_analyze, "qv_mi", "Query-Value")
    plot_QKV_head_mi(QKV_mi_list, analyzer.N_attention_layers, min_val, max_val, N_src_tokens, epochs_to_analyze, "kv_mi", "Key-Value")

