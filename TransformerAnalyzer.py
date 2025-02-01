# Implementation of the TransformerAnalyzer class

import numpy as np
from pathlib import Path
import os
from typing import Tuple
import torch
from tqdm import tqdm

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer

# Definitions of the classes implemented by this application
from ProbeManager import ProbeManager


class TransformerAnalyzer:
    # Constructor
    def __init__(self, config : dict, probe_config : dict) -> None:
        super().__init__()
        # Get the model configuration parameters
        self._seq_len = config["seq_len"]
        self._d_model = config["d_model"]
        self._N_epochs = config["num_epochs"]
        self._probe_dir = Path(config["analyze_dir"])
        self._lang_src = config["lang_src"]
        self._lang_tgt = config["lang_tgt"]
        self._tokenizer_dir = config["tokenizer_dir"]

        # Contains the probe file names
        self._probe_config = probe_config


    def __load_tokenizers(self):
        # Load the source and target language tokenizers from the JSON file created
        # during the training of the model
        tokenizer_src_fname = Path(f"{self._tokenizer_dir}/tokenizer_{self._lang_src}.json")
        self._tokenizer_src = Tokenizer.from_file(str(tokenizer_src_fname))

        tokenizer_tgt_fname = Path(f"{self._tokenizer_dir}/tokenizer_{self._lang_tgt}.json")
        self._tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_fname))        


    def __init_probes(self):
        # Encoder probe objects
        self._enc_embedding_probe = ProbeManager()          # Encoder's embedding layer probe
        self._enc_0_attn_probe = ProbeManager()             # Encoder 0 attention layer probe
        self._enc_0_feedforward_probe = ProbeManager()      # Encoder 0 feedforward layer probe
        self._enc_5_attn_probe = ProbeManager()             # Encoder 5 attention layer probe
        self._enc_5_feedforward_probe = ProbeManager()      # Encoder 5 feedforward layer probe
        self._encoder_probe = ProbeManager()                # Encoder block's input and output probe

        # Decoder probe objects
        self._dec_embedding_probe = ProbeManager()          # Decoder's embedding layer probe
        self._dec_0_attn_probe = ProbeManager()             # Decoder 0 attention layer probe
        self._dec_0_cross_attn_probe = ProbeManager()       # Decoder 0 cross-attention layer probe
        self._dec_0_feedforward_probe = ProbeManager()      # Decoder 0 feedforward layer probe
        self._dec_5_attn_probe = ProbeManager()             # Decoder 5 attention layer probe
        self._dec_5_cross_attn_probe = ProbeManager()       # Decoder 5 cross-attention layer probe
        self._dec_5_feedforward_probe = ProbeManager()      # Decoder 5 feedforward layer probe
        self._decoder_probe = ProbeManager()                # Decoder block's input and output probe

        # Projection layer probe
        self._projection_probe = ProbeManager()             # Projection probe object 


    def __load_probes(self, epoch) -> None:
        # Load the epoch's encoder probes
        self._enc_embedding_probe.load(epoch, self._probe_dir, self._probe_config["enc_embed_layer"])
        self._enc_0_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_0_attn"])
        self._enc_0_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_0_feedforward"])
        self._enc_5_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_5_attn"])
        self._enc_5_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_5_feedforward"])
        self._encoder_probe.load(epoch, self._probe_dir, self._probe_config["enc_block"]) 

        # Load the epoch's decoder probes
        self._dec_embedding_probe.load(epoch, self._probe_dir, self._probe_config["dec_embed_layer"])
        self._dec_0_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_attn"])
        self._dec_0_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_cross_attn"])
        self._dec_0_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_feedforward"])
        self._dec_5_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_attn"])
        self._dec_5_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_cross_attn"])
        self._dec_5_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_feedforward"])
        self._decoder_probe.load(epoch, self._probe_dir, self._probe_config["dec_block"])

        # Load the epoch's projection layer probe
        self._projection_probe.load(epoch, self._probe_dir, self._probe_config["proj_layer"])

        
    def __analyze_probes(self) -> None:
        N_inputs = len(self._encoder_probe._probe_in)

        for i in range(N_inputs):
            # Get the source sentence mask and number of tokens
            # from the encoder block's input probe
            src_mask=self._encoder_probe._probe_in[i]["mask"]
            N_tokens = np.count_nonzero(np.squeeze(src_mask))

            # From the encoder's embedding layer input probe get the source and target tokens
            enc_embedding_input = self._enc_embedding_probe._probe_in[i]
            src_tokens = enc_embedding_input["src_tokens"]
            src_sentence_tokens = np.squeeze(src_tokens)[:N_tokens]
            print(f"Source sentence: {self._tokenizer_src.decode(src_sentence_tokens)}")

            tgt_mask=enc_embedding_input["tgt_mask"]
            N_words = np.count_nonzero(np.squeeze(tgt_mask))
            tgt_tokens = enc_embedding_input["tgt_tokens"]
            tgt_sentence_tokens = np.squeeze(tgt_tokens)[:N_words]
            print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")


    def run(self) -> None:
        if self._probe_dir.exists():
            if self._probe_dir.is_dir() == False:
                raise ValueError(f"Invalid probe directory name: {str(self._probe_dir)}")
        else:
            raise RuntimeError(f"Probe directory {str(self._probe_dir)} does not exist")

        # Load the Tokenizer data from the files that were created during model training
        self.__load_tokenizers()

        # Create the ProbeManager objects
        self.__init_probes()
            
        for epoch in range(self._N_epochs):
            # For this epoch, load all the probe files from disk
            self.__load_probes(epoch)
            self.__analyze_probes()




