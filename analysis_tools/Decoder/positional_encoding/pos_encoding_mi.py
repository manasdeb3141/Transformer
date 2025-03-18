
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

import torch
import torch.nn as nn
import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mutual_info_estimator import MutualInfoEstimator

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer

def rotary_position_embedding(max_seq_len, dim):
    # Calculate the angle rates based on dimension indices.
    angle_rates = 1 / torch.pow(10000, torch.arange(0, dim, 2).float() / dim)
    # Calculate the angles for each position for half of the dimensions (sine and cosine)
    angles = (torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0))
    # Cosines and sines of the angles to get the RoPE for each position
    position_encodings = torch.stack((angles.cos(), angles.sin()), dim=2).flatten(1)
    return position_encodings

def apply_rope_embeddings(embeddings, position_encodings):
    # Split the position encodings into cosines and sines
    cos_enc, sin_enc = position_encodings[..., 0::2], position_encodings[..., 1::2]
    # Apply the rotations
    embeddings[..., 0::2] = embeddings[..., 0::2] * cos_enc - embeddings[..., 1::2] * sin_enc
    embeddings[..., 1::2] = embeddings[..., 1::2] * cos_enc + embeddings[..., 0::2] * sin_enc
    return embeddings 

def plot_decoder_embedding_MI(pos_encoding_MI, ROPE_pos_encoding_MI, epoch):
    # Plot the MI values for the positional encoding outputs        
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Mutual information of the Positional Encoding layer output for epoch {epoch}")
    for i in range(4):
        a = i//2
        b = i%2
        im = axs[a, b].imshow(pos_encoding_MI[i]["MI"], cmap=plt.cm.Wistia)
        # im = axs[a, b].imshow(self.__enc_embedding_MI[i]["MI"], cmap=plt.cm.Set1)
        axs[a, b].set_xticks(range(0, len(pos_encoding_MI[i]["input_words"])), pos_encoding_MI[i]["input_words"], rotation=90)
        axs[a, b].set_yticks(range(0, len(pos_encoding_MI[i]["input_words"])), pos_encoding_MI[i]["input_words"], rotation=0)
        axs[a, b].set_title(f"Mutual information: sentence {i}")

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show(block=True)

    # Plot the MI values for the ROPE positional encoding outputs
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Mutual information of the ROPE Positional Encoding layer output for epoch {epoch}")
    for i in range(4):
        a = i//2
        b = i%2
        im = axs[a, b].imshow(ROPE_pos_encoding_MI[i]["MI"], cmap=plt.cm.Wistia)
        # im = axs[a, b].imshow(self.__enc_embedding_MI[i]["MI"], cmap=plt.cm.Set1)
        axs[a, b].set_xticks(range(0, len(ROPE_pos_encoding_MI[i]["input_words"])), ROPE_pos_encoding_MI[i]["input_words"], rotation=90)
        axs[a, b].set_yticks(range(0, len(ROPE_pos_encoding_MI[i]["input_words"])), ROPE_pos_encoding_MI[i]["input_words"], rotation=0)
        axs[a, b].set_title(f"Mutual information: sentence {i}")

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show(block=True)

class PELayerAnalyzer(TransformerAnalyzer) :
    def __init__(self, model_config: dict, probe_config: dict) -> None:
        super().__init__(model_config, probe_config)
        self._MI_estimator = MutualInfoEstimator()

    def apply_ROPE_pos_encoding(self, dec_embedding_output, N_tokens, tgt_sentence_tokens):
        max_seq_len = 350
        dim = 512

        # Generate the position encodings for the sequence
        PE_array = rotary_position_embedding(dec_embedding_output.shape[0], dim)

        # Apply the RoPE to the token embeddings
        ROPE_pos_encoding_tensor = apply_rope_embeddings(torch.from_numpy(dec_embedding_output), PE_array)
        ROPE_pos_encoding = ROPE_pos_encoding_tensor.numpy()

        N_valid_tokens = N_tokens
        x = np.arange(0, N_valid_tokens, 1)
        y = np.arange(0, N_valid_tokens, 1)
        x_pos, y_pos = np.meshgrid(x, y)
        xy_pos = np.vstack([x_pos.ravel(), y_pos.ravel()]).T

        input_words = list()

        for token in tgt_sentence_tokens:
            input_words.append(super().get_tgt_word_from_token(token))

        MI = np.zeros((N_valid_tokens, N_valid_tokens))
        for x, y in xy_pos:
            X = ROPE_pos_encoding[x]
            Y = ROPE_pos_encoding[y]
            self._MI_estimator.set_inputs(X, Y)
            MI_data = self._MI_estimator.kraskov_MI()
            # _, MI_data = self._MI_estimator.kernel_MI()
            MI[x, y] = MI_data["MI"]

        MI_dict = dict(input_words=input_words, x_pos=x_pos, y_pos=y_pos, MI=MI)
        return MI_dict

    def __analyze_pos_encoding(self, decoder_input : np.array, N_tokens : int, tgt_sentence_tokens : np.array) -> None:
        # Contains the actual number of tokens that will be processed
        N_valid_tokens = N_tokens

        # Flag to ignore the SOS and EOS tokens
        ignore_SOS_EOS = False

        if ignore_SOS_EOS:
            # Ignore the SOS token at the beginning of the sentence
            if tgt_sentence_tokens[0] == 2:
                offset = 1
                N_valid_tokens -= 1
            else:
                offset = 0

            # Ignore the EOS token at the end of the sentence
            if tgt_sentence_tokens[N_tokens-1] == 3:
                N_valid_tokens -= 1
        else:
            offset = 0
        
        x = np.arange(0, N_valid_tokens, 1)
        y = np.arange(0, N_valid_tokens, 1)
        x_pos, y_pos = np.meshgrid(x, y)
        xy_pos = np.vstack([x_pos.ravel(), y_pos.ravel()]).T

        input_words = list()
        for token in tgt_sentence_tokens:
            if ignore_SOS_EOS:
                if token > 3:
                    input_words.append(super().get_tgt_word_from_token(token))
            else:
                input_words.append(super().get_tgt_word_from_token(token))

        MI = np.zeros((N_valid_tokens, N_valid_tokens))
        for x, y in xy_pos:
            X = decoder_input[x+offset]
            Y = decoder_input[y+offset]
            self._MI_estimator.set_inputs(X, Y)
            MI_data = self._MI_estimator.kraskov_MI()
            # _, MI_data = self._MI_estimator.kernel_MI()
            MI[x, y] = MI_data["MI"]

        MI_dict = dict(input_words=input_words, x_pos=x_pos, y_pos=y_pos, MI=MI)
        return MI_dict



    def __process_probes(self):
        print("Analyzing the Positional Encoder Layer output of the Decoder")

        epoch = 19
        sentence_id = 3

        # Load all the decoder embedding layer probe files from disk
        super().load_dec_embedding_probes(epoch)

        # Number of input sentences in this epoch
        N_inputs = len(self.decoder_probe._probe_in)

        # Initialize the list of mutual information values for the encoder embedding layer 
        pos_encoding_MI = list()
        ROPE_pos_encoding_MI = list()

        # Each decoder probe is a list of lists with each element corresponding to a decoder token
        N_decoder_probes = len(self.decoder_probe._probe_in[sentence_id])

        # Token ids that we are interested in
        token_id_list = [4, 6, 8, N_decoder_probes-1]

        for token_id in token_id_list:
            # Get the number of tokens in the source sentence input to the encoder
            src_mask=self.decoder_probe._probe_in[sentence_id][token_id]["src_mask"]
            N_src_tokens = np.count_nonzero(np.squeeze(src_mask))

            # Get the number of tokens in the sentence input to the decoder
            tgt_mask=self.decoder_probe._probe_in[sentence_id][token_id]["tgt_mask"]
            N_tgt_tokens = tgt_mask.shape[1]

            # Get the decoder's embedding layer input and output probes
            dec_embedding_input = self.dec_embedding_probe._probe_in[sentence_id][token_id]
            dec_embedding_output = self.dec_embedding_probe._probe_out[sentence_id][token_id]

            # From the decoder's embedding layer input probe get the target tokens
            tgt_sentence_tokens = np.squeeze(dec_embedding_input)[:N_tgt_tokens]
            # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

            decoder_input = self.decoder_probe._probe_in[sentence_id][token_id]["decoder_in"]
            MI_dict = self.__analyze_pos_encoding(np.squeeze(decoder_input), N_tgt_tokens, tgt_sentence_tokens)
            pos_encoding_MI.append(MI_dict)

            # Apply the ROPE positional encoding to the embedded layer output and analyze the MI
            ROPE_MI_dict = self.apply_ROPE_pos_encoding(np.squeeze(dec_embedding_output), N_tgt_tokens, tgt_sentence_tokens)
            ROPE_pos_encoding_MI.append(ROPE_MI_dict)

        plot_decoder_embedding_MI(pos_encoding_MI, ROPE_pos_encoding_MI, epoch)


    def run(self):
        super().run()
        self.__process_probes()


def main():
    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../../../model_data/opus_books_en_fr/tokens"
    model_config["analyze_dir"] = "../../../model_data/opus_books_en_fr/probes_8"

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Analyze the Transformer's encoder emebdding layer probes
    analyzer = PELayerAnalyzer(model_config, probe_config)
    analyzer.run()

    input("Press any key to continue...")

# Entry point of the program
if __name__ == '__main__':
    main()