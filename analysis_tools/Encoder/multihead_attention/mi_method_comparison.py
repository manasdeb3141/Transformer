
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

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer
from mutual_info_estimator import MutualInfoEstimator

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_FF import get_FF_input_output
from stack_matrices import stack_FF_matrix

def extract_attention_layer_data(ff_stacked_dict, attention_layer, dimension_to_extract):
    match attention_layer:
        case 0:
            ff_in = ff_stacked_dict["ff_0"]["ff_in"]
            ff_out = ff_stacked_dict["ff_0"]["ff_out"]

        case 1:
            ff_in = ff_stacked_dict["ff_1"]["ff_in"]
            ff_out = ff_stacked_dict["ff_1"]["ff_out"]

        case 2:
            ff_in = ff_stacked_dict["ff_2"]["ff_in"]
            ff_out = ff_stacked_dict["ff_2"]["ff_out"]

        case 3:
            ff_in = ff_stacked_dict["ff_3"]["ff_in"]
            ff_out = ff_stacked_dict["ff_3"]["ff_out"]

        case 4:
            ff_in = ff_stacked_dict["ff_4"]["ff_in"]
            ff_out = ff_stacked_dict["ff_4"]["ff_out"]

        case 5:
            ff_in = ff_stacked_dict["ff_5"]["ff_in"]
            ff_out = ff_stacked_dict["ff_5"]["ff_out"]

        case 6:
            ff_in = ff_stacked_dict["ff_6"]["ff_in"]
            ff_out = ff_stacked_dict["ff_6"]["ff_out"]

    X = ff_in[:, dimension_to_extract]
    Y = ff_out[:, dimension_to_extract]

    # Instantiate the Mutual Information Estimator object
    # and get the probability and mutual information dictionaries
    MI_estimator = MutualInfoEstimator(X, Y)

    # Get probabilities and MI estimates from the KDE estimator
    prob_data, KDE_MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
    KDE_MI = KDE_MI_data['MI']

    # Get the MI estimate from the Kraskov estimator
    Kraskov_MI_data = MI_estimator.kraskov_MI()
    Kraskov_MI = Kraskov_MI_data['MI']

    # Get the MINE MI estimate
    MINE_MI, _ = MI_estimator.MINE_MI()

    P_XY = prob_data['P_XY']
    x_grid = prob_data['x_grid']
    y_grid = prob_data['y_grid']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, P_XY, cmap=plt.cm.jet)
    fig.colorbar(surf, ax=ax)
    ax.set_xlabel('X (input)')
    ax.set_ylabel('Y (output)')
    ax.set_title(f"Joint PDF P(X,Y) of the Encoder feed forward layer 5")
    ax.text(-0.7475, 15, 0.0006, f"KSG MI: {Kraskov_MI:.2f} bits\nMINE MI: {MINE_MI:.2f} bits\nKDE MI: {KDE_MI:.2f} bits", color='red', fontsize=12)

    plt.show(block=True)


def extract_prob_data(analyzer):
    epoch_to_analyze = 19
    attention_layer = 5
    dimension_to_extract = 347

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

    extract_attention_layer_data(ff_stacked_dict, attention_layer, dimension_to_extract)


def main():
    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../model_data/opus_books_en_fr/tokens"
    model_config["analyze_dir"] = "../../model_data/opus_books_en_fr/probes_8"

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Instantiate the Transformer's analyzer object.
    # Load the tokenizer and instantiate the ProbeManager objects
    analyzer = TransformerAnalyzer(model_config, probe_config)
    analyzer.run()

    # Run the requested test
    extract_prob_data(analyzer)


# Entry point of the program
if __name__ == '__main__':
    main()
