
import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer
from mutual_info_estimator import MutualInfoEstimator

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_FF import get_FF_input_output


def plot_FF_mi(FF_mi, epoch, encoder_layer, sentence_id):
    FF_MI_estimate = FF_mi["FF_MI_estimate"]
    input_words = FF_mi["input_words"]

    fig, ax = plt.subplots()
    # img = ax.imshow(FF_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    img = ax.imshow(FF_MI_estimate.T, cmap=plt.cm.Wistia)
    ax.set_aspect('equal')
    ax.set_title(f"MI between the input and output of the FF network")
    ax.set_xticks(range(0, len(input_words)), input_words, rotation=90)
    ax.set_yticks(range(0, len(input_words)), input_words, rotation=0)

    fig.colorbar(img, ax=ax)
    fig.suptitle(f"Mutual Information between the input and output of the FF network\nfor epoch {epoch}, attention layer {encoder_layer}, sentence {sentence_id}")
    plt.show(block=True)


def compute_FF_mi(analyzer, FF_inout, encoder_layer, sentence_tokens):
    ff_dict = FF_inout[f"ff_{encoder_layer}"]
    ff_in = ff_dict["ff_in"]
    ff_out = ff_dict["ff_out"]

    # Dimensions of the MI matrix will be N_rows x N_rows
    N_rows = ff_in.shape[0] 
    FF_MI_estimate = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between input and output of the FF network...")
    for i, j in tqdm(ij_pos):
        X = ff_in[i]
        Y = ff_out[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        FF_MI_estimate[i, j] = MI

    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    FF_mi_dict = dict(FF_MI_estimate=FF_MI_estimate, input_words=input_words)

    return FF_mi_dict

# Main function of this script
def main():
    print("Computing the mutual information between the inputs and outputs of the FF network...")

    epoch = 19
    encoder_layer = 2
    sentence_id = 3

    save_file = Path(f"data/ff_mi_epoch_{epoch}_layer_{encoder_layer}_sentence_{sentence_id}.pt")

    if save_file.exists():
        print(f"FF mutual information file {str(save_file)} found. Loading it ...")
        FF_mi = torch.load(save_file, weights_only=False)
    else:
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

        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch)
        
        # Number of input sentences in this epoch
        N_inputs = len(analyzer.encoder_probe._probe_in)

        # Get the tokens of the source and target sentences
        N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)

        # Get the query, key, value arrays for all the attention layers of this input sentence
        FF_inout = get_FF_input_output(analyzer, sentence_id, N_src_tokens)
        FF_mi = compute_FF_mi(analyzer, FF_inout, encoder_layer, src_sentence_tokens)

        # Save the file
        torch.save(FF_mi, save_file)
    
    plot_FF_mi(FF_mi, epoch, encoder_layer, sentence_id)


# Entry point of the script
if __name__ == '__main__':
    main()