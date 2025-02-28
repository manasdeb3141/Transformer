
import sys
sys.path.append('../..')
sys.path.append('../utils')
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
from get_QKV import get_query_key_value_matrix
from get_sentence_tokens import get_sentence_tokens

def process_q_prime(analyzer, QKV_list, attention_layer, sentence_id):
    QKV_dict = QKV_list[sentence_id]
    QKV_atten_dict = QKV_dict[f"attention_{attention_layer}"]
    sentence_tokens = QKV_dict["sentence_tokens"]
    x = QKV_atten_dict["x"]
    query = QKV_atten_dict["query"]
    key = QKV_atten_dict["key"]
    value = QKV_atten_dict["value"]

    N_rows = query.shape[0] 

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between the rows of Q_prime ...")
    # Initialize the mutual information matrix
    query_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = query[i]
        Y_row = query[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        query_MI_estimate[i, j] = MI


    print("\nComputing the mutual information between the rows of K_prime ...")
    # Initialize the mutual information matrix
    key_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = key[i]
        Y_row = key[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        key_MI_estimate[i, j] = MI

    print("\nComputing the mutual information between the rows of V_prime ...")
    # Initialize the mutual information matrix
    value_MI_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(ij_pos):
        X_row = value[i]
        Y_row = value[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        value_MI_estimate[i, j] = MI

    
    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    np.fill_diagonal(query_MI_estimate, 0)
    np.fill_diagonal(key_MI_estimate, 0)
    np.fill_diagonal(value_MI_estimate, 0)

    fig, axs = plt.subplots(1,3)

    img = axs[0].imshow(query_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[0].set_aspect('auto')
    axs[0].set_title(f"Mutual Information between the rows of Q_prime")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(key_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[1].set_aspect('auto')
    axs[1].set_title(f"Mutual Information between the rows of K_prime")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(value_MI_estimate.T, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    axs[2].set_aspect('auto')
    axs[2].set_title(f"Mutual Information between the rows of V_prime")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    fig.colorbar(img, ax=axs[2])

    plt.show(block=True)



# Main function of this script
def main():
    # Check if the GPU is available
    # cuda_is_avail = check_system()
    # device = torch.device("cuda:0" if cuda_is_avail else "cpu")

    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../model_data/opus_books_en_fr/tokens"
    model_config["model_dir"] = "../../model_data/opus_books_en_fr/weights"
    model_config["analyze_dir"] = "../../model_data/opus_books_en_fr/probes_8"
    #model_config["tokenizer_dir"] = "../../model_data_d32/opus_books_en_fr/tokens"
    #model_config["model_dir"] = "../../model_data_d32/opus_books_en_fr/weights"

    epoch = 19
    attention_layer = 2
    analyze_sentence = 3

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Instantiate the Transformer's analyzer object.
    # Load the tokenizer and instantiate the ProbeManager objects
    analyzer = TransformerAnalyzer(model_config, probe_config)
    analyzer.run()

    # For this epoch, load all the encoder layer probe files from disk
    analyzer.load_encoder_probes(epoch)

    # Number of input sentences in this epoch
    N_inputs = len(analyzer.encoder_probe._probe_in)

    # This will contain the QKV dictionaries for all the attention layers
    # of all the input sentences of this epoch
    QKV_list = list()

    for sentence_id in range(N_inputs):
        N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)

        # Get the query, key, value arrays for all the attention layers of this input sentence
        QKV_dict = get_query_key_value_matrix(analyzer, sentence_id, N_src_tokens)
        QKV_dict["sentence_tokens"] = src_sentence_tokens
        QKV_list.append(QKV_dict)

    process_q_prime(analyzer, QKV_list, attention_layer, analyze_sentence)

# Entry point of the script
if __name__ == '__main__':
    main()