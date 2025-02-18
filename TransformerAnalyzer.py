# Implementation of the TransformerAnalyzer class

import sys
sys.path.append('./utils')

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
from mutual_info_estimator import MutualInfoEstimator


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

        # Function to analyze the Transformer probes. This is populated
        # by the class that inherits this class
        self._test_id = None
        self._analyze_probes = None

    def __load_tokenizers(self):
        # Load the source and target language tokenizers from the JSON file created
        # during the training of the model
        tokenizer_src_fname = Path(f"{self._tokenizer_dir}/tokenizer_{self._lang_src}.json")
        self._tokenizer_src = Tokenizer.from_file(str(tokenizer_src_fname))

        # Get the source language vocabulary as a dictionary
        src_vocab = self._tokenizer_src.get_vocab()
        self._src_vocab_keys = list(src_vocab.keys())
        self._src_vocab_values = list(src_vocab.values())

        tokenizer_tgt_fname = Path(f"{self._tokenizer_dir}/tokenizer_{self._lang_tgt}.json")
        self._tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_fname))        


    def __init_probes(self):
        # Encoder probe objects
        self._enc_embedding_probe = ProbeManager()          # Encoder's embedding layer probe

        self._enc_0_attn_probe = ProbeManager()             # Encoder 0 attention layer probe
        self._enc_0_feedforward_probe = ProbeManager()      # Encoder 0 feedforward layer probe

        self._enc_1_attn_probe = ProbeManager()             # Encoder 1 attention layer probe
        self._enc_1_feedforward_probe = ProbeManager()      # Encoder 1 feedforward layer probe

        self._enc_2_attn_probe = ProbeManager()             # Encoder 2 attention layer probe
        self._enc_2_feedforward_probe = ProbeManager()      # Encoder 2 feedforward layer probe

        self._enc_3_attn_probe = ProbeManager()             # Encoder 3 attention layer probe
        self._enc_3_feedforward_probe = ProbeManager()      # Encoder 3 feedforward layer probe

        self._enc_4_attn_probe = ProbeManager()             # Encoder 4 attention layer probe
        self._enc_4_feedforward_probe = ProbeManager()      # Encoder 4 feedforward layer probe

        self._enc_5_attn_probe = ProbeManager()             # Encoder 5 attention layer probe
        self._enc_5_feedforward_probe = ProbeManager()      # Encoder 5 feedforward layer probe

        self._encoder_probe = ProbeManager()                # Encoder block's input and output probe

        # Decoder probe objects
        self._dec_embedding_probe = ProbeManager()          # Decoder's embedding layer probe

        self._dec_0_attn_probe = ProbeManager()             # Decoder 0 attention layer probe
        self._dec_0_cross_attn_probe = ProbeManager()       # Decoder 0 cross-attention layer probe
        self._dec_0_feedforward_probe = ProbeManager()      # Decoder 0 feedforward layer probe

        self._dec_1_attn_probe = ProbeManager()             # Decoder 1 attention layer probe
        self._dec_1_cross_attn_probe = ProbeManager()       # Decoder 1 cross-attention layer probe
        self._dec_1_feedforward_probe = ProbeManager()      # Decoder 1 feedforward layer probe

        self._dec_2_attn_probe = ProbeManager()             # Decoder 2 attention layer probe
        self._dec_2_cross_attn_probe = ProbeManager()       # Decoder 2 cross-attention layer probe
        self._dec_2_feedforward_probe = ProbeManager()      # Decoder 2 feedforward layer probe

        self._dec_3_attn_probe = ProbeManager()             # Decoder 3 attention layer probe
        self._dec_3_cross_attn_probe = ProbeManager()       # Decoder 3 cross-attention layer probe
        self._dec_3_feedforward_probe = ProbeManager()      # Decoder 3 feedforward layer probe

        self._dec_4_attn_probe = ProbeManager()             # Decoder 4 attention layer probe
        self._dec_4_cross_attn_probe = ProbeManager()       # Decoder 4 cross-attention layer probe
        self._dec_4_feedforward_probe = ProbeManager()      # Decoder 4 feedforward layer probe

        self._dec_5_attn_probe = ProbeManager()             # Decoder 5 attention layer probe
        self._dec_5_cross_attn_probe = ProbeManager()       # Decoder 5 cross-attention layer probe
        self._dec_5_feedforward_probe = ProbeManager()      # Decoder 5 feedforward layer probe

        self._decoder_probe = ProbeManager()                # Decoder block's input and output probe

        # Projection layer probe
        self._projection_probe = ProbeManager()             # Projection probe object 


    def get_src_word_from_token(self, token_id : int) -> str:
        return self._src_vocab_keys[self._src_vocab_values.index(token_id)]


    def load_enc_embedding_probes(self, epoch, load_epoch=True) -> None:
        self._enc_embedding_probe.load(epoch, self._probe_dir, self._probe_config["enc_embed_layer"], load_epoch)
        self._encoder_probe.load(epoch, self._probe_dir, self._probe_config["enc_block"], load_epoch) 

    
    def load_encoder_probes(self, epoch, load_epoch=True) -> None:
        self._enc_embedding_probe.load(epoch, self._probe_dir, self._probe_config["enc_embed_layer"], load_epoch)
        self._enc_0_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_0_attn"], load_epoch)
        self._enc_0_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_0_feedforward"], load_epoch)
        self._enc_1_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_1_attn"], load_epoch)
        self._enc_1_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_1_feedforward"], load_epoch)
        self._enc_2_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_2_attn"], load_epoch)
        self._enc_2_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_2_feedforward"], load_epoch)
        self._enc_3_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_3_attn"], load_epoch)
        self._enc_3_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_3_feedforward"], load_epoch)
        self._enc_4_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_4_attn"], load_epoch)
        self._enc_4_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_4_feedforward"], load_epoch)
        self._enc_5_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_5_attn"], load_epoch)
        self._enc_5_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_5_feedforward"], load_epoch)
        self._encoder_probe.load(epoch, self._probe_dir, self._probe_config["enc_block"], load_epoch)


    def load_decoder_probes(self, epoch, load_epoch=True) -> None:
        self._dec_embedding_probe.load(epoch, self._probe_dir, self._probe_config["dec_embed_layer"], load_epoch)
        self._dec_0_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_attn"], load_epoch)
        self._dec_0_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_cross_attn"], load_epoch)
        self._dec_0_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_feedforward"], load_epoch)
        self._dec_1_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_1_attn"], load_epoch)
        self._dec_1_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_1_cross_attn"], load_epoch)
        self._dec_1_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_1_feedforward"], load_epoch)
        self._dec_2_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_2_attn"], load_epoch)
        self._dec_2_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_2_cross_attn"], load_epoch)
        self._dec_2_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_2_feedforward"], load_epoch)
        self._dec_3_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_3_attn"], load_epoch)
        self._dec_3_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_3_cross_attn"], load_epoch)
        self._dec_3_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_3_feedforward"], load_epoch)
        self._dec_4_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_4_attn"], load_epoch)
        self._dec_4_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_4_cross_attn"], load_epoch)
        self._dec_4_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_4_feedforward"], load_epoch)
        self._dec_5_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_attn"], load_epoch)
        self._dec_5_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_cross_attn"], load_epoch)
        self._dec_5_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_feedforward"], load_epoch)
        self._decoder_probe.load(epoch, self._probe_dir, self._probe_config["dec_block"], load_epoch)


    def load_probes(self, epoch, load_epoch=True) -> None:
        # Load the epoch's encoder probes
        self.load_encoder_probes(epoch, load_epoch)

        # Load the epoch's decoder probes
        self.load_decoder_probes(epoch, load_epoch)

        # Load the epoch's projection layer probe
        self._projection_probe.load(epoch, self._probe_dir, self._probe_config["proj_layer"], load_epoch)


    def run(self, **kwargs) -> None:
        if self._probe_dir.exists():
            if self._probe_dir.is_dir() == False:
                raise ValueError(f"Invalid probe directory name: {str(self._probe_dir)}")
        else:
            raise RuntimeError(f"Probe directory {str(self._probe_dir)} does not exist")

        # Load the Tokenizer data from the files that were created during model training
        self.__load_tokenizers()

        # Create the ProbeManager objects
        self.__init_probes()

        # Create the Mutual Information Estimator object
        self._MI_estimator = MutualInfoEstimator()
            
        # The member variable _analyze_probes is set by the class that inherits this class
        if self._analyze_probes is not None:
            if "test" in kwargs:
                self._test_id = kwargs["test"]

            self._analyze_probes()

 




