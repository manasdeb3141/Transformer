# Function to compute the entropy and mutual information of the input and output of the feedforwrd layer
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
from get_FF import get_FF_input_output
from stack_matrices import stack_FF_matrix
from compute_matrix_pdf import compute_ff_pdf

def plot_feed_forward_pdf_single_epoch(ff_pdf_list : list, attention_layer : int):
    
    MI_list = list()
    max_index = 0
    max_MI = -1e9
    P_XY = None

    for index, pdf_entry in enumerate(ff_pdf_list):
        prob_data = pdf_entry[0]
        MI_data = pdf_entry[1]
        MI = MI_data["MI"]
        MI_list.append(MI)

        if MI > max_MI:
            max_MI = MI
            max_index = index
            P_XY = prob_data['P_XY']
            x_grid = prob_data['x_grid']
            y_grid = prob_data['y_grid']
        

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    x_axis = np.arange(0, len(MI_list))
    ax.bar(x_axis, MI_list)
    ax.set_title(f"Mutual Information of FF Layer {attention_layer} over dimensions")
    ax.set_xlabel("Dimension (d_model)")
    # ax.set_xlabel("Sequence length")
    ax.set_ylabel("Mutual Information")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, P_XY, cmap=plt.cm.jet)
    fig.colorbar(surf, ax=ax)
    ax.set_xlabel('X (input)')
    ax.set_ylabel('Y (output)')
    ax.set_title(f"Joint PDF surface P(X,Y) for the Feedforward layer (index: {max_index})")

    plt.show(block=False)


def process_feed_forward_single_epoch(analyzer : TransformerAnalyzer):
    print("Computing the entropy and mutual information between the input and output of the FF layer ...")
    ff_pdf_filename = 'data/ff_pdf_data.pt'

    epoch_to_analyze = 19
    attention_layer = 5

    ff_pdf_list = None
    ff_pdf_file = Path(ff_pdf_filename)
    if ff_pdf_file.is_file():
        print("Feedforward pdf list file found. Loading it ...")
        ff_pdf_list = torch.load(ff_pdf_file, weights_only=False)

    if ff_pdf_list is None:
        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch_to_analyze)

        # Number of input sentences in this epoch
        N_inputs = len(analyzer.encoder_probe._probe_in)

        # This will contain the ff dictionaries of all the input sentences 
        # of this epoch. Each dictionary will contain the input and output
        # of the FF layer 
        ff_list = list()

        # Iterate across all the input sentences of this epoch and get the query, key and value arrays.
        # Stack the arrays horizontally after each iteration
        for i in range(N_inputs):
            N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, i)
            # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
            # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")
            ff_dict = get_FF_input_output(analyzer, i, N_src_tokens)
            ff_list.append(ff_dict)

        # Concatenate the input and output matrices of the FF layer horizontally 
        # for all the input sentences of this epoch
        ff_stacked_dict = stack_FF_matrix(ff_list, N_inputs)

        ff_pdf_list = compute_ff_pdf(ff_stacked_dict, attention_layer)

        # Save the pdf of the input and output of the FF layer
        # torch.save(ff_pdf_list, ff_pdf_filename)

    # Plot the pdf of the input and output of the FF layer
    plot_feed_forward_pdf_single_epoch(ff_pdf_list, attention_layer)

def plot_feed_forward_pdf(ff_mi_list : list, attention_layer : int):
    mi_image_array = None

    for mi_list_entry in ff_mi_list:
        mi_list = np.array(mi_list_entry)

        if mi_image_array is None:
            mi_image_array = mi_list
        else:
            mi_image_array = np.vstack((mi_image_array, mi_list))

    epochs = np.arange(0, mi_image_array.shape[0], 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(mi_image_array.T, cmap=plt.cm.jet, origin='lower', extent=[0, mi_image_array.shape[0]-1, 0, mi_image_array.shape[1]-1])
    #im = ax.imshow(mi_image_array.T, cmap=plt.cm.jet, vmin=0, vmax=4, origin='lower', extent=[0, mi_image_array.shape[0]-1, 0, mi_image_array.shape[1]-1])
    ax.set_aspect('auto')
    ax.set_title(f"Mutual Information of Feedforward Layer {attention_layer} over dimensions")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Mutual Information")
    ax.set_xticks(range(0, len(epochs)), epochs)

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show(block=False)

def extract_ff_mi(ff_pdf_list):
    MI_list = list()
    max_index = 0
    max_MI = -1e9
    max_MI_prob_data = None
    P_XY = None

    for pdf_entry in ff_pdf_list:
        prob_data = pdf_entry[0]
        MI_data = pdf_entry[1]
        MI = MI_data["MI"]
        MI_list.append(MI)

        if MI > max_MI:
            max_MI = MI
            max_MI_prob_data = prob_data
        
    return max_MI, MI_list, max_MI_prob_data


def process_feed_forward_attn_layer(analyzer : TransformerAnalyzer, attention_layer : int):
    print(f"Computing the entropy and mutual information between the input and output of the FF layer {attention_layer} ...")

    epochs_to_analyze = np.arange(0, 20, 1)
    ff_mi_filename = f'data/ff_mi_attn_layer_{attention_layer}_data.pt'

    ff_mi_list = None
    ff_mi_file = Path(ff_mi_filename)
    if ff_mi_file.is_file():
        print("Feedforward MI list file found. Loading it ...")
        ff_mi_dict = torch.load(ff_mi_file, weights_only=False)
        ff_mi_list = ff_mi_dict["ff_mi_list"]

    if ff_mi_list is None:
        # This will contain the MI values across dimensions for each epoch. 
        # It is a list of lists where each entry is a list containing the MI values
        # for each dimension for an epoch
        ff_mi_list = list()

        max_MI_pdf_data = None
        max_MI = -1e9

        # Analyze the probes of the Multihead Attention layer of the encoder for each epoch
        for epoch in epochs_to_analyze:
            print(f"Analyzing epoch {epoch} ...")
            # For this epoch, load all the encoder layer probe files from disk
            analyzer.load_encoder_probes(epoch)

            # Number of input sentences in this epoch
            N_inputs = len(analyzer.encoder_probe._probe_in)

            # This will contain the ff dictionaries of all the input sentences 
            # of this epoch. Each dictionary will contain the input and output
            # of the FF layer 
            ff_list = list()

            # Iterate across all the input sentences of this epoch and get the query, key and value arrays.
            # Stack the arrays horizontally after each iteration
            for i in range(N_inputs):
                N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, i)
                # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")
                # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")
                ff_dict = get_FF_input_output(analyzer, i, N_src_tokens)
                ff_list.append(ff_dict)

            # Concatenate the input and output matrices of the FF layer horizontally 
            # for all the input sentences of this epoch
            ff_stacked_dict = stack_FF_matrix(ff_list, N_inputs)

            # Compute the MI values across the d_model dimension for the input and 
            # output of the FF layer
            ff_pdf_list = compute_ff_pdf(ff_stacked_dict, attention_layer)

            # Extract the MI from the list of dictionaries. The prob_data dictionary
            # of the maximum MI is returned
            largest_MI, MI_list, prob_data = extract_ff_mi(ff_pdf_list)

            if largest_MI > max_MI:
                max_MI = largest_MI
                max_MI_pdf_data = prob_data

            # Append the MI values across the d_model dimension for this epoch
            ff_mi_list.append(MI_list)

        # Save the pdf of the input and output of the FF layer
        ff_mi_dict = dict(ff_mi_list=ff_mi_list, max_MI=max_MI, max_MI_pdf_data=max_MI_pdf_data)
        torch.save(ff_mi_dict, ff_mi_filename)

    # Plot the pdf of the input and output of the FF layer
    plot_feed_forward_pdf(ff_mi_list, attention_layer)

def process_feed_forward(analyzer : TransformerAnalyzer):
    for i in range(6):
        process_feed_forward_attn_layer(analyzer, i)