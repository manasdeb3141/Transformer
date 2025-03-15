
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
import shutil
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

from utils import user_confirmation

def process_qkv_prime(analyzer, QKV_dict, attention_layer):
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

    qkv_prime_dict = \
        dict(query_MI_estimate=query_MI_estimate, 
             key_MI_estimate=key_MI_estimate, 
             value_MI_estimate=value_MI_estimate,
             input_words=input_words)

    return qkv_prime_dict


def process_qkv_prime_symmetric(analyzer, QKV_dict, attention_layer):
    QKV_atten_dict = QKV_dict[f"attention_{attention_layer}"]
    sentence_tokens = QKV_dict["sentence_tokens"]
    x = QKV_atten_dict["x"]
    query = QKV_atten_dict["query"]
    key = QKV_atten_dict["key"]
    value = QKV_atten_dict["value"]

    N_rows = query.shape[0] 
    temp = np.triu_indices(N_rows, k=0)
    upper_idx = np.vstack((temp[0], temp[1])).T

    print("\nComputing the mutual information between the rows of Q_prime ...")

    # Initialize the mutual information and JSD matrix
    query_MI_estimate = np.zeros((N_rows, N_rows))
    query_JSD_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(upper_idx):
        X_row = query[i]
        Y_row = query[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        prob_data, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        P_X = prob_data["P_X"]
        P_Y = prob_data["P_Y"]
        JSD = MI_estimator.JS_divergence(P_X, P_Y)
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        query_MI_estimate[i, j] = MI
        query_JSD_estimate[i, j] = JSD

    query_MI_estimate = query_MI_estimate + query_MI_estimate.T - np.diag(np.diag(query_MI_estimate))
    query_JSD_estimate = query_JSD_estimate + query_JSD_estimate.T - np.diag(np.diag(query_JSD_estimate))

    print("\nComputing the mutual information between the rows of K_prime ...")

    # Initialize the mutual information matrix
    key_MI_estimate = np.zeros((N_rows, N_rows))
    key_JSD_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(upper_idx):
        X_row = key[i]
        Y_row = key[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        prob_data, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        P_X = prob_data["P_X"]
        P_Y = prob_data["P_Y"]
        JSD = MI_estimator.JS_divergence(P_X, P_Y)
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        key_MI_estimate[i, j] = MI
        key_JSD_estimate[i, j] = JSD

    key_MI_estimate = key_MI_estimate + key_MI_estimate.T - np.diag(np.diag(key_MI_estimate))
    key_JSD_estimate = key_JSD_estimate + key_JSD_estimate.T - np.diag(np.diag(key_JSD_estimate))

    print("\nComputing the mutual information between the rows of V_prime ...")

    # Initialize the mutual information matrix
    value_MI_estimate = np.zeros((N_rows, N_rows))
    value_JSD_estimate = np.zeros((N_rows, N_rows))

    for i, j in tqdm(upper_idx):
        X_row = value[i]
        Y_row = value[j]

        MI_estimator = MutualInfoEstimator(X_row, Y_row)
        prob_data, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        P_X = prob_data["P_X"]
        P_Y = prob_data["P_Y"]
        JSD = MI_estimator.JS_divergence(P_X, P_Y)
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
        value_MI_estimate[i, j] = MI
        value_JSD_estimate[i, j] = JSD

    value_MI_estimate = value_MI_estimate + value_MI_estimate.T - np.diag(np.diag(value_MI_estimate))
    value_JSD_estimate = value_JSD_estimate + value_JSD_estimate.T - np.diag(np.diag(value_JSD_estimate))
    
    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    qkv_prime_dict = \
        dict(query_MI_estimate=query_MI_estimate, 
             key_MI_estimate=key_MI_estimate, 
             value_MI_estimate=value_MI_estimate,
             query_JSD_estimate=query_JSD_estimate,
             key_JSD_estimate=key_JSD_estimate,
             value_JSD_estimate=value_JSD_estimate,
             input_words=input_words)

    return qkv_prime_dict


# Main function of this script
def main():
    # Directory where the extracted data will be saved
    save_data_dir = Path("data/qkv_prime/model_data")

    if save_data_dir.exists():
        if (user_confirmation(f"The directory {save_data_dir} already exists. Do you want to delete it?")): 
            # First delete and then create the directory
            shutil.rmtree(save_data_dir)
            save_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_data_dir.mkdir(parents=True, exist_ok=True)
        
    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../../model_data/opus_books_en_fr/tokens"
    model_config["model_dir"] = "../../../model_data/opus_books_en_fr/weights"
    model_config["analyze_dir"] = "../../../model_data/opus_books_en_fr/probes_8"
    #model_config["tokenizer_dir"] = "../../../model_data_d32/opus_books_en_fr/tokens"
    #model_config["model_dir"] = "../../../model_data_d32/opus_books_en_fr/weights"

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Instantiate the Transformer's analyzer object.
    # Load the tokenizer and instantiate the ProbeManager objects
    analyzer = TransformerAnalyzer(model_config, probe_config)
    analyzer.run()

    # Parameters of the extraction
    epochs_to_analyze = np.arange(0, 20, 1)
    attention_layers = [0, 1, 3, 5]

    for epoch in epochs_to_analyze:
        print(f"Processing epoch {epoch} ...")

        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch)

        # Number of input sentences in this epoch
        N_inputs = len(analyzer.encoder_probe._probe_in)

        for sentence_id in range(N_inputs):
            if sentence_id != 3:
                continue

            # This will contain the QKV_prime dictionaries for all the attention layers
            QKV_list = list()

            print(f"Processing sentence {sentence_id} ...")

            save_file = save_data_dir / f"qkv_prime_epoch_{epoch:02d}_sentence_{sentence_id:02d}.pt"
            if save_file.exists():
                continue

            N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)

            # Get the query, key, value arrays for all the attention layers of this input sentence
            QKV_dict = get_query_key_value_matrix(analyzer, sentence_id, N_src_tokens)
            QKV_dict["sentence_tokens"] = src_sentence_tokens

            for attention_layer in attention_layers:
                qkv_prime_dict = process_qkv_prime_symmetric(analyzer, QKV_dict, attention_layer)
                QKV_list.append(qkv_prime_dict)

            # Save the QKV_prime list to disk
            torch.save(QKV_list, save_file)


# Entry point of the script
if __name__ == '__main__':
    main()