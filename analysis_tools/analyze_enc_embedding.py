
import sys
sys.path.append('..')
sys.path.append('../utils')

import torch
import torch.nn as nn
import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer

class EmbeddingLayerAnalyzer(TransformerAnalyzer) :
    def __init__(self, model_config: dict, probe_config: dict) -> None:
        super().__init__(model_config, probe_config)

        # Set the probe analysis function callback
        # member variable inherited from the parent class
        self._analyze_probes = self.__process_probes


    def __analyze_enc_embedding(self, sentence_id : int, enc_embedding_output : np.array, N_tokens : int, src_sentence_tokens : np.array) -> None:
        # Contains the actual number of tokens that will be processed
        N_valid_tokens = N_tokens

        # Ignore the BOS token at the beginning of the sentence
        if (src_sentence_tokens[0] == 2):
            offset = 1
            N_valid_tokens -= 1
        else:
            offset = 0

        if (src_sentence_tokens[N_tokens-1] == 3):
            N_valid_tokens -= 1
        
        x = np.arange(0, N_valid_tokens, 1)
        y = np.arange(0, N_valid_tokens, 1)
        x_pos, y_pos = np.meshgrid(x, y)
        xy_pos = np.vstack([x_pos.ravel(), y_pos.ravel()]).T

        input_words = list()
        for token in src_sentence_tokens:
            if (token > 3):
                input_words.append(super().get_src_word_from_token(token))

        MI = np.zeros((N_valid_tokens, N_valid_tokens))
        for x, y in xy_pos:
            X = enc_embedding_output[x+offset]
            Y = enc_embedding_output[y+offset]
            self._MI_estimator.set_inputs(X, Y)
            MI_data = self._MI_estimator.kraskov_MI()
            MI[x, y] = MI_data["MI"]

        MI_dict = dict(input_words=input_words, x_pos=x_pos, y_pos=y_pos, MI=MI)
        self.__enc_embedding_MI.append(MI_dict)
        
    def __process_probes(self):
        print("Analyzing the Embedding Layer")

        # For the encoder embedding layer analyze the last epoch
        epoch = self._N_epochs-1

        # For this epoch, load all the probe files from disk
        super().load_enc_embedding_probes(epoch)

        # Number of input sentences in this epoch
        N_inputs = len(self._encoder_probe._probe_in)

        # Initialize the list of mutual information values for the encoder embedding layer 
        self.__enc_embedding_MI = list()

        # Iterate across all the input sentences of this epoch
        for i in range(N_inputs):
            # Get the number of tokens in this source sentence
            # from the encoder block's input mask probe
            src_mask=self._encoder_probe._probe_in[i]["mask"]
            N_tokens = np.count_nonzero(np.squeeze(src_mask))

            # Get the encoder's embedding layer input and output probes
            enc_embedding_input = self._enc_embedding_probe._probe_in[i]
            enc_embedding_output = self._enc_embedding_probe._probe_out[i]

            # From the encoder's embedding layer input probe get the source and target tokens
            src_tokens = enc_embedding_input["src_tokens"]
            src_sentence_tokens = np.squeeze(src_tokens)[:N_tokens]
            # print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")

            tgt_mask=enc_embedding_input["tgt_mask"]
            N_words = np.count_nonzero(np.squeeze(tgt_mask))
            tgt_tokens = enc_embedding_input["tgt_tokens"]
            tgt_sentence_tokens = np.squeeze(tgt_tokens)[:N_words]
            # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

            self.__analyze_enc_embedding(i, np.squeeze(enc_embedding_output), N_tokens, src_sentence_tokens)

        # Plot the MI values for the encoder embedding layer        
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Mutual Information of the Encoder embedding layer output vectors")
        for i in range(4):
            a = i//2
            b = i%2
            im = axs[a, b].imshow(self.__enc_embedding_MI[i]["MI"], cmap=plt.cm.Wistia)
            axs[a, b].set_xticks(range(0, len(self.__enc_embedding_MI[i]["input_words"])), self.__enc_embedding_MI[i]["input_words"], rotation=45)
            axs[a, b].set_yticks(range(0, len(self.__enc_embedding_MI[i]["input_words"])), self.__enc_embedding_MI[i]["input_words"], rotation=45)
            axs[a, b].set_title(f"Mutual information: sentence {i}")

        plt.subplots_adjust(hspace=0.8, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()


def main():
    # Get the model configuration
    cfg_obj = LangModelConfig()
    model_config = cfg_obj.get_config()

    model_config["tokenizer_dir"] = "../model_data/opus_books_en_fr/tokens"
    model_config["analyze_dir"] = "../model_data/opus_books_en_fr/probes_4"

    # Dictionary of probe file names
    probe_config = cfg_obj.get_probes()

    # Analyze the Transformer's encoder emebdding layer probes
    analyzer = EmbeddingLayerAnalyzer(model_config, probe_config)
    analyzer.run()


# Entry point of the program
if __name__ == '__main__':
    main()