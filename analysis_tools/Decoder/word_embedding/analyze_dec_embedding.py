
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

def plot_decoder_embedding_MI(dec_embedding_MI, epoch):
    # Plot the MI values for the decoder embedding layer        
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Mutual information of the decoder embedding layer output for epoch {epoch}")

    for i in range(4):
        a = i//2
        b = i%2
        im = axs[a, b].imshow(dec_embedding_MI[i]["MI"], cmap=plt.cm.Wistia)
        axs[a, b].set_xticks(range(0, len(dec_embedding_MI[i]["input_words"])), dec_embedding_MI[i]["input_words"], rotation=90)
        axs[a, b].set_yticks(range(0, len(dec_embedding_MI[i]["input_words"])), dec_embedding_MI[i]["input_words"], rotation=0)
        axs[a, b].set_title(f"Mutual information: input token #{dec_embedding_MI[i]["token_idx"]}")

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show(block=False)

class EmbeddingLayerAnalyzer(TransformerAnalyzer) :
    def __init__(self, model_config: dict, probe_config: dict) -> None:
        super().__init__(model_config, probe_config)
        self._MI_estimator = MutualInfoEstimator()


    def __analyze_dec_embedding(self, dec_embedding_output : np.array, N_tokens : int, src_sentence_tokens : np.array) -> None:
        # Contains the actual number of tokens that will be processed
        N_valid_tokens = N_tokens

        # Flag to ignore the SOS and EOS tokens
        ignore_SOS_EOS = False

        if ignore_SOS_EOS:
            # Ignore the SOS token at the beginning of the sentence
            if src_sentence_tokens[0] == 2:
                offset = 1
                N_valid_tokens -= 1
            else:
                offset = 0

            # Ignore the EOS token at the end of the sentence
            if src_sentence_tokens[N_tokens-1] == 3:
                N_valid_tokens -= 1
        else:
            offset = 0
        
        x = np.arange(0, N_valid_tokens, 1)
        y = np.arange(0, N_valid_tokens, 1)
        x_pos, y_pos = np.meshgrid(x, y)
        xy_pos = np.vstack([x_pos.ravel(), y_pos.ravel()]).T

        input_words = list()
        for token in src_sentence_tokens:
            if ignore_SOS_EOS:
                if token > 3:
                    input_words.append(super().get_tgt_word_from_token(token))
            else:
                input_words.append(super().get_tgt_word_from_token(token))

        MI = np.zeros((N_valid_tokens, N_valid_tokens))
        for x, y in xy_pos:
            X = dec_embedding_output[x+offset]
            Y = dec_embedding_output[y+offset]
            self._MI_estimator.set_inputs(X, Y)
            MI_data = self._MI_estimator.kraskov_MI()
            # _, MI_data = self._MI_estimator.kernel_MI(KDE_module='sklearn')
            MI[x, y] = MI_data["MI"]

        MI_dict = dict(input_words=input_words, x_pos=x_pos, y_pos=y_pos, MI=MI)
        return MI_dict
        
    def __process_probes(self):
        print("Analyzing the Embedding Layer")

        epochs_to_analyze = [0, 4, 9, 14, 19]
        epochs_to_analyze = [19]

        # Analyze the probes of each epoch of the decoder embedding layer
        for epoch in epochs_to_analyze:
            # For this epoch, load all the decoder embedding layer probe files from disk
            super().load_dec_embedding_probes(epoch)

            # Number of input sentences in this epoch
            N_inputs = len(self.decoder_probe._probe_in)

            # Iterate across all the input sentences of this epoch
            for i in range(N_inputs):
                if i not in [3]:
                    continue

                # Each decoder probe is a list of lists with each element corresponding to a decoder token
                N_decoder_probes = len(self.decoder_probe._probe_in[i])

                # Initialize the list of mutual information values for the decoder embedding layer 
                dec_embedding_MI = list()

                for probe_idx in range(N_decoder_probes):
                    if probe_idx not in [2, 4, 6, N_decoder_probes-1]:
                        continue

                    # Get the number of tokens in the source sentence input to the encoder
                    src_mask=self.decoder_probe._probe_in[i][probe_idx]["src_mask"]
                    N_src_tokens = np.count_nonzero(np.squeeze(src_mask))

                    # Get the number of tokens in the sentence input to the decoder
                    tgt_mask=self.decoder_probe._probe_in[i][probe_idx]["tgt_mask"]
                    N_tgt_tokens = tgt_mask.shape[1]

                    # Get the decoder's embedding layer input and output probes
                    dec_embedding_input = self.dec_embedding_probe._probe_in[i][probe_idx]
                    dec_embedding_output = self.dec_embedding_probe._probe_out[i][probe_idx]

                    # From the decoder's embedding layer input probe get the target tokens
                    tgt_sentence_tokens = np.squeeze(dec_embedding_input)[:N_tgt_tokens]
                    # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

                    MI_dict = self.__analyze_dec_embedding(np.squeeze(dec_embedding_output), N_tgt_tokens, tgt_sentence_tokens)
                    MI_dict["token_idx"] = probe_idx+1
                    dec_embedding_MI.append(MI_dict)

                plot_decoder_embedding_MI(dec_embedding_MI, epoch)


    def run(self):
        super().run()
        self.__process_probes()


def main():
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

    # Analyze the Transformer's decoder emebdding layer probes
    analyzer = EmbeddingLayerAnalyzer(model_config, probe_config)
    analyzer.run()

    input("Press any key to continue...")

# Entry point of the program
if __name__ == '__main__':
    main()