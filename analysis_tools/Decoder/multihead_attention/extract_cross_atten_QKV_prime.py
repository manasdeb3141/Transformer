

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
        MI = MI_data["MI"]
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
        #MI_data = MI_estimator.kraskov_MI()
        MI = MI_data["MI"]
        # MI, _ = MI_estimator.MINE_MI()
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

    # Get the target words from the tokens
    input_words = list()
    for token in tgt_sentence_tokens:
            input_words.append(analyzer.get_tgt_word_from_token(token))

    return dict(input_words=input_words, QQ_MI_estimate=QQ_MI_estimate, QK_MI_estimate=QK_MI_estimate, QV_MI_estimate=QV_MI_estimate)

def extract_cross_atten_QKV_prime(epoch, attention_layer, sentence_id, decoder_token_id):
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

    save_file = Path(f"data/cross_QKV/QKV_cross_epoch_{epoch}_layer_{attention_layer}_sentence_{sentence_id}_token_{decoder_token_id}.pt")

    if save_file.exists():
        print(f"QKV cross-attention data file {str(save_file)} found. Loading it ...")
        cross_QKV_prime_save_dict = torch.load(save_file, weights_only=False)
        QKV_MI_dict = cross_QKV_prime_save_dict["QKV_MI_dict"]
        cross_QKV_prime_MI_dict = cross_QKV_prime_save_dict["cross_QKV_prime_MI_dict"]
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

        QKV_dict = get_cross_atten_QKV(analyzer, sentence_id, decoder_token_id, N_src_tokens)

        QKV_MI_dict = process_cross_atten_QKV_inputs(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens)
        cross_QKV_prime_MI_dict = process_cross_atten_QKV_prime(analyzer, QKV_dict, attention_layer, tgt_sentence_tokens)

        cross_QKV_prime_save_dict = dict(QKV_MI_dict=QKV_MI_dict, cross_QKV_prime_MI_dict=cross_QKV_prime_MI_dict)

        # Save the file
        torch.save(cross_QKV_prime_save_dict, save_file)


def get_max_tokens(epoch, attention_layer, sentence_id):
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

    # For this epoch, load all the decoder layer probe files from disk
    analyzer.load_decoder_probes(epoch)

    # Get the number of tokens in the sentence input to the decoder
    num_tokens=len(analyzer.decoder_probe._probe_in[sentence_id])
    tgt_mask=analyzer.decoder_probe._probe_in[sentence_id][num_tokens-1]["tgt_mask"]

    return num_tokens-1, tgt_mask.shape[0]


def main():
    # Analysis parameters
    epoch = 19
    attention_layer = 0
    sentence_id = 3

    max_tokens, _ = get_max_tokens(epoch, attention_layer, sentence_id)
    print("Max tokens: ", max_tokens)

    token_list = np.arange(10, max_tokens)

    for decoder_token_id in tqdm(token_list.tolist()):
        extract_cross_atten_QKV_prime(epoch, attention_layer, sentence_id, decoder_token_id)

# Entry point of the program
if __name__ == '__main__':
    main()