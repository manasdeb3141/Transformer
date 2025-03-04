
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
from get_sentence_tokens import get_encoder_sentence_tokens
from get_FF import get_FF_input_output

def process_encoder_decoder_outputs(analyzer, encoder_output, decoder_output, projection_input, enc_sentence_tokens, dec_sentence_tokens):
    # Dimensions of the MI matrix will be N_rows x N_rows
    N_rows = len(enc_sentence_tokens)
    enc_MI_estimate = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between encoder output rows ...")
    for i, j in tqdm(ij_pos):
        X = encoder_output[i]
        Y = encoder_output[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        enc_MI_estimate[i, j] = MI


    # Dimensions of the MI matrix will be N_rows x N_rows
    N_rows = decoder_output.shape[0] 
    dec_MI_estimate = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between decoder output rows ...")
    for i, j in tqdm(ij_pos):
        X = decoder_output[i]
        Y = decoder_output[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        dec_MI_estimate[i, j] = MI

    enc_input_words = list()
    for token in enc_sentence_tokens:
        enc_input_words.append(analyzer.get_src_word_from_token(token))

    dec_input_words = list()
    for token in dec_sentence_tokens:
        dec_input_words.append(analyzer.get_tgt_word_from_token(token))

    return dict(enc_MI_estimate=enc_MI_estimate, dec_MI_estimate=dec_MI_estimate, enc_input_words=enc_input_words, dec_input_words=dec_input_words)


def plot_encoder_decoder_MI(MI_dict, epoch, sentence_id):
    enc_MI_estimate = MI_dict["enc_MI_estimate"]
    dec_MI_estimate = MI_dict["dec_MI_estimate"]
    enc_input_words = MI_dict["enc_input_words"]
    dec_input_words = MI_dict["dec_input_words"]

    fig, axs = plt.subplots(1, 2)
    img = axs[0].imshow(enc_MI_estimate.T, cmap=plt.cm.Wistia)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between the encoder output rows")
    axs[0].set_xticks(range(0, len(enc_input_words)), enc_input_words, rotation=90)
    axs[0].set_yticks(range(0, len(enc_input_words)), enc_input_words, rotation=0)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(dec_MI_estimate.T, cmap=plt.cm.Wistia)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"MI between the decoder output rows")
    axs[1].set_xticks(range(0, len(dec_input_words)), dec_input_words, rotation=90)
    axs[1].set_yticks(range(0, len(dec_input_words)), dec_input_words, rotation=0)
    fig.colorbar(img, ax=axs[1])

    fig.suptitle(f"Mutual Information between the encoder and decoder output rows\nfor epoch {epoch}, sentence {sentence_id}")
    plt.show(block=True)


# Main function of this script
def main():
    print("Computing the mutual information between the inputs and outputs of the projection layer ...")

    epoch = 19
    sentence_id = 3
    decoder_token_id = 10

    save_file = Path(f"data/projection_mi_epoch_{epoch}_sentence_{sentence_id}_token_{decoder_token_id}.pt")

    if save_file.exists():
        print(f"Projection layer mutual information file {str(save_file)} found. Loading it ...")
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

        # For this epoch, load all the probe files from disk
        analyzer.load_probes(epoch)
        
        # Number of input sentences in this epoch
        N_inputs = len(analyzer.decoder_probe._probe_in)

        # Get the tokens of the source and target sentences
        _, _, enc_sentence_tokens = get_encoder_sentence_tokens(analyzer, sentence_id)
        _, _, dec_sentence_tokens = get_sentence_tokens(analyzer, sentence_id, decoder_token_id)

        # Get the input and output arrays of the projection layer for this input sentence and token
        projection_input = analyzer.projection_probe._probe_in[sentence_id][decoder_token_id]
        projection_output = analyzer.projection_probe._probe_out[sentence_id][decoder_token_id]
        encoder_output = analyzer.encoder_probe._probe_out[sentence_id]
        decoder_output = analyzer.decoder_probe._probe_out[sentence_id][decoder_token_id]
        # decoder_cross_attn_output = analyzer.dec_5_cross_attn_probe._probe_out[sentence_id][decoder_token_id]

        print(f"Projection input shape: {projection_input.shape}")
        print(f"Projection output shape: {projection_output.shape}")
        print(f"Decoder output shape: {decoder_output.shape}")

        MI_dict = process_encoder_decoder_outputs(analyzer, 
                                                  encoder_output.squeeze(), 
                                                  decoder_output.squeeze(), 
                                                  projection_input.squeeze(), 
                                                  enc_sentence_tokens,
                                                  dec_sentence_tokens)

        plot_encoder_decoder_MI(MI_dict, epoch, sentence_id)


        # Save the file
        # torch.save(FF_mi, save_file)
    


# Entry point of the script
if __name__ == '__main__':
    main()