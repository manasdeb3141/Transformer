# Implementation of the TransformerAnalyzer class

import sys

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

        # Number of attention layers in the encoder and decoder
        self.N_attention_layers = 6

        # Contains the probe file names
        self._probe_config = probe_config

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

        # Get the target language vocabulary as a dictionary
        tgt_vocab = self._tokenizer_tgt.get_vocab()
        self._tgt_vocab_keys = list(tgt_vocab.keys())
        self._tgt_vocab_values = list(tgt_vocab.values())

    def __init_probes(self):
        # Encoder probe objects
        self.enc_embedding_probe = ProbeManager()          # Encoder's embedding layer probe

        self.enc_0_attn_probe = ProbeManager()             # Encoder 0 attention layer probe
        self.enc_0_feedforward_probe = ProbeManager()      # Encoder 0 feedforward layer probe

        self.enc_1_attn_probe = ProbeManager()             # Encoder 1 attention layer probe
        self.enc_1_feedforward_probe = ProbeManager()      # Encoder 1 feedforward layer probe

        self.enc_2_attn_probe = ProbeManager()             # Encoder 2 attention layer probe
        self.enc_2_feedforward_probe = ProbeManager()      # Encoder 2 feedforward layer probe

        self.enc_3_attn_probe = ProbeManager()             # Encoder 3 attention layer probe
        self.enc_3_feedforward_probe = ProbeManager()      # Encoder 3 feedforward layer probe

        self.enc_4_attn_probe = ProbeManager()             # Encoder 4 attention layer probe
        self.enc_4_feedforward_probe = ProbeManager()      # Encoder 4 feedforward layer probe

        self.enc_5_attn_probe = ProbeManager()             # Encoder 5 attention layer probe
        self.enc_5_feedforward_probe = ProbeManager()      # Encoder 5 feedforward layer probe

        self.encoder_probe = ProbeManager()                # Encoder block's input and output probe

        # Decoder probe objects
        self.dec_embedding_probe = ProbeManager()          # Decoder's embedding layer probe

        self.dec_0_attn_probe = ProbeManager()             # Decoder 0 attention layer probe
        self.dec_0_cross_attn_probe = ProbeManager()       # Decoder 0 cross-attention layer probe
        self.dec_0_feedforward_probe = ProbeManager()      # Decoder 0 feedforward layer probe

        self.dec_1_attn_probe = ProbeManager()             # Decoder 1 attention layer probe
        self.dec_1_cross_attn_probe = ProbeManager()       # Decoder 1 cross-attention layer probe
        self.dec_1_feedforward_probe = ProbeManager()      # Decoder 1 feedforward layer probe

        self.dec_2_attn_probe = ProbeManager()             # Decoder 2 attention layer probe
        self.dec_2_cross_attn_probe = ProbeManager()       # Decoder 2 cross-attention layer probe
        self.dec_2_feedforward_probe = ProbeManager()      # Decoder 2 feedforward layer probe

        self.dec_3_attn_probe = ProbeManager()             # Decoder 3 attention layer probe
        self.dec_3_cross_attn_probe = ProbeManager()       # Decoder 3 cross-attention layer probe
        self.dec_3_feedforward_probe = ProbeManager()      # Decoder 3 feedforward layer probe

        self.dec_4_attn_probe = ProbeManager()             # Decoder 4 attention layer probe
        self.dec_4_cross_attn_probe = ProbeManager()       # Decoder 4 cross-attention layer probe
        self.dec_4_feedforward_probe = ProbeManager()      # Decoder 4 feedforward layer probe

        self.dec_5_attn_probe = ProbeManager()             # Decoder 5 attention layer probe
        self.dec_5_cross_attn_probe = ProbeManager()       # Decoder 5 cross-attention layer probe
        self.dec_5_feedforward_probe = ProbeManager()      # Decoder 5 feedforward layer probe

        self.decoder_probe = ProbeManager()                # Decoder block's input and output probe

        # Projection layer probe
        self.projection_probe = ProbeManager()             # Projection probe object 

    def load_enc_embedding_probes(self, epoch, load_epoch=True) -> None:
        self.enc_embedding_probe.load(epoch, self._probe_dir, self._probe_config["enc_embed_layer"], load_epoch)
        self.encoder_probe.load(epoch, self._probe_dir, self._probe_config["enc_block"], load_epoch) 

    def load_encoder_probes(self, epoch, load_epoch=True) -> None:
        self.enc_embedding_probe.load(epoch, self._probe_dir, self._probe_config["enc_embed_layer"], load_epoch)
        self.enc_0_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_0_attn"], load_epoch)
        self.enc_0_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_0_feedforward"], load_epoch)
        self.enc_1_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_1_attn"], load_epoch)
        self.enc_1_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_1_feedforward"], load_epoch)
        self.enc_2_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_2_attn"], load_epoch)
        self.enc_2_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_2_feedforward"], load_epoch)
        self.enc_3_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_3_attn"], load_epoch)
        self.enc_3_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_3_feedforward"], load_epoch)
        self.enc_4_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_4_attn"], load_epoch)
        self.enc_4_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_4_feedforward"], load_epoch)
        self.enc_5_attn_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_5_attn"], load_epoch)
        self.enc_5_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["enc_layer_5_feedforward"], load_epoch)
        self.encoder_probe.load(epoch, self._probe_dir, self._probe_config["enc_block"], load_epoch)

    def load_dec_embedding_probes(self, epoch, load_epoch=True) -> None:
        self.dec_embedding_probe.load(epoch, self._probe_dir, self._probe_config["dec_embed_layer"], load_epoch)
        self.decoder_probe.load(epoch, self._probe_dir, self._probe_config["dec_block"], load_epoch) 

    def load_decoder_probes(self, epoch, load_epoch=True) -> None:
        self.dec_embedding_probe.load(epoch, self._probe_dir, self._probe_config["dec_embed_layer"], load_epoch)
        self.dec_0_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_attn"], load_epoch)
        self.dec_0_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_cross_attn"], load_epoch)
        self.dec_0_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_0_feedforward"], load_epoch)
        self.dec_1_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_1_attn"], load_epoch)
        self.dec_1_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_1_cross_attn"], load_epoch)
        self.dec_1_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_1_feedforward"], load_epoch)
        self.dec_2_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_2_attn"], load_epoch)
        self.dec_2_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_2_cross_attn"], load_epoch)
        self.dec_2_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_2_feedforward"], load_epoch)
        self.dec_3_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_3_attn"], load_epoch)
        self.dec_3_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_3_cross_attn"], load_epoch)
        self.dec_3_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_3_feedforward"], load_epoch)
        self.dec_4_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_4_attn"], load_epoch)
        self.dec_4_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_4_cross_attn"], load_epoch)
        self.dec_4_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_4_feedforward"], load_epoch)
        self.dec_5_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_attn"], load_epoch)
        self.dec_5_cross_attn_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_cross_attn"], load_epoch)
        self.dec_5_feedforward_probe.load(epoch, self._probe_dir, self._probe_config["dec_layer_5_feedforward"], load_epoch)
        self.decoder_probe.load(epoch, self._probe_dir, self._probe_config["dec_block"], load_epoch)

    def load_projection_probes(self, epoch, load_epoch=True) -> None:
        # Load the epoch's projection layer probe
        self.projection_probe.load(epoch, self._probe_dir, self._probe_config["proj_layer"], load_epoch)

    def load_probes(self, epoch, load_epoch=True) -> None:
        # Load the epoch's encoder probes
        self.load_encoder_probes(epoch, load_epoch)

        # Load the epoch's decoder probes
        self.load_decoder_probes(epoch, load_epoch)

        # Load the epoch's projection layer probe
        self.projection_probe.load(epoch, self._probe_dir, self._probe_config["proj_layer"], load_epoch)

    def get_src_word_from_token(self, token_id : int) -> str:
        return self._src_vocab_keys[self._src_vocab_values.index(token_id)]

    def get_tgt_word_from_token(self, token_id : int) -> str:
        return self._tgt_vocab_keys[self._tgt_vocab_values.index(token_id)]

    def get_encoder_input_token_count(self, sentence_id : int) -> int:
        # Get the number of tokens in this source sentence
        # from the encoder block's input mask probe
        input_mask=self.encoder_probe._probe_in[sentence_id]["mask"]
        N_input_tokens = np.count_nonzero(np.squeeze(input_mask))

        return N_input_tokens

    def get_encoder_input_tokens(self, sentence_id : int) -> dict:
        # Get the number of tokens in this source sentence
        # from the encoder block's input mask probe
        input_mask=self.encoder_probe._probe_in[sentence_id]["mask"]
        N_input_tokens = np.count_nonzero(np.squeeze(input_mask))

        # Get the encoder's embedding layer input probes
        enc_embedding_input = self.enc_embedding_probe._probe_in[sentence_id]

        # From the encoder's embedding layer input probe get the source tokens
        input_tokens_full = enc_embedding_input["src_tokens"]
        input_tokens = np.squeeze(input_tokens_full)[:N_input_tokens]

        return N_input_tokens, input_tokens

    def get_encoder_input_words(self, sentence_id : int) -> list:
        _, input_tokens = self.get_encoder_input_tokens(sentence_id)

        input_words = list()
        for token in input_tokens:
            input_words.append(self.get_src_word_from_token(token))

        return input_words

    def get_encoder_QKV(self, enc_layer : int, sentence_id : int, N_rows : int) -> dict:
        # Get the Q, K, and V input matrices and the Q', K', V' matrices
        #  for the specific encoder layer for the specific input sentence
        match enc_layer:
            case 0:
                enc_attn_input = self.enc_0_attn_probe._probe_in[sentence_id]
            case 1:
                enc_attn_input = self.enc_1_attn_probe._probe_in[sentence_id]
            case 2:
                enc_attn_input = self.enc_2_attn_probe._probe_in[sentence_id]
            case 3:
                enc_attn_input = self.enc_3_attn_probe._probe_in[sentence_id]
            case 4:
                enc_attn_input = self.enc_4_attn_probe._probe_in[sentence_id]
            case 5:
                enc_attn_input = self.enc_5_attn_probe._probe_in[sentence_id]
            case _:
                raise ValueError(f"Invalid encoder layer: {enc_layer}")

        Q_full = enc_attn_input["q"].squeeze()
        K_full = enc_attn_input["k"].squeeze()
        V_full = enc_attn_input["v"].squeeze()
        Q_prime_full = enc_attn_input["query"].squeeze()
        K_prime_full = enc_attn_input["key"].squeeze()
        V_prime_full = enc_attn_input["value"].squeeze()

        Q = Q_full[:N_rows]
        K = K_full[:N_rows]
        V = V_full[:N_rows]
        Q_prime = Q_prime_full[:N_rows]
        K_prime = K_prime_full[:N_rows]
        V_prime = V_prime_full[:N_rows]

        QKV_dict = dict(Q=Q, K=K, V=V, Q_prime=Q_prime, K_prime=K_prime, V_prime=V_prime)
        return QKV_dict

    def get_encoder_QKV_head(self, enc_layer : int, head_id : int, sentence_id : int, N_rows : int) -> dict:
        # Get the Q, K, and V head matrices and scored V matrix
        #  for the specific encoder layer for the specific input sentence
        match enc_layer:
            case 0:
                enc_attn_input = self.enc_0_attn_probe._probe_in[sentence_id]
            case 1:
                enc_attn_input = self.enc_1_attn_probe._probe_in[sentence_id]
            case 2:
                enc_attn_input = self.enc_2_attn_probe._probe_in[sentence_id]
            case 3:
                enc_attn_input = self.enc_3_attn_probe._probe_in[sentence_id]
            case 4:
                enc_attn_input = self.enc_4_attn_probe._probe_in[sentence_id]
            case 5:
                enc_attn_input = self.enc_5_attn_probe._probe_in[sentence_id]
            case _:
                raise ValueError(f"Invalid encoder layer: {enc_layer}")

        Q_head_full = enc_attn_input["query_head"].squeeze()
        K_head_full = enc_attn_input["key_head"].squeeze()
        V_head_full = enc_attn_input["value_head"].squeeze()
        V_scored_full = enc_attn_input["scored_value"].squeeze()

        Q_heads = Q_head_full[:,:N_rows]
        K_heads = K_head_full[:,:N_rows]
        V_heads = V_head_full[:,:N_rows]
        V_scored_heads = V_scored_full[:,:N_rows]

        QKV_head_dict = dict(Q_head=Q_heads[head_id], K_head=K_heads[head_id], V_head=V_heads[head_id], V_scored_head=V_scored_heads[head_id])
        return QKV_head_dict

    def get_encoder_attention_scores(self, enc_layer : int, head : int, sentence_id : int, N_rows : int) -> np.array:
        match enc_layer:
            case 0:
                enc_attn_input = self.enc_0_attn_probe._probe_in[sentence_id]

            case 1:
                enc_attn_input = self.enc_1_attn_probe._probe_in[sentence_id]

            case 2:
                enc_attn_input = self.enc_2_attn_probe._probe_in[sentence_id]

            case 3:
                enc_attn_input = self.enc_3_attn_probe._probe_in[sentence_id]

            case 4:
                enc_attn_input = self.enc_4_attn_probe._probe_in[sentence_id]

            case 5:
                enc_attn_input = self.enc_5_attn_probe._probe_in[sentence_id]

            case _:
                raise ValueError(f"Invalid encoder layer: {enc_layer}")

        all_head_attention_scores = enc_attn_input["attention_scores"].squeeze()
        attention_scores = all_head_attention_scores[head]

        return attention_scores[:N_rows, :N_rows]

    def get_encoder_feedforward_inout(self, enc_layer : int, sentence_id : int, N_rows : int) -> dict:
        match enc_layer:
            case 0:
                enc_ff_input = self.enc_0_feedforward_probe._probe_in[sentence_id]
                enc_ff_output = self.enc_0_feedforward_probe._probe_out[sentence_id]

            case 1:
                enc_ff_input = self.enc_1_feedforward_probe._probe_in[sentence_id]
                enc_ff_output = self.enc_1_feedforward_probe._probe_out[sentence_id]

            case 2:
                enc_ff_input = self.enc_2_feedforward_probe._probe_in[sentence_id]
                enc_ff_output = self.enc_2_feedforward_probe._probe_out[sentence_id]

            case 3:
                enc_ff_input = self.enc_3_feedforward_probe._probe_in[sentence_id]
                enc_ff_output = self.enc_3_feedforward_probe._probe_out[sentence_id]

            case 4:
                enc_ff_input = self.enc_4_feedforward_probe._probe_in[sentence_id]
                enc_ff_output = self.enc_4_feedforward_probe._probe_out[sentence_id]

            case 5:
                enc_ff_input = self.enc_5_feedforward_probe._probe_in[sentence_id]
                enc_ff_output = self.enc_5_feedforward_probe._probe_out[sentence_id]

            case _:
                raise ValueError(f"Invalid encoder layer: {enc_layer}")
            
        ff_in_full = enc_ff_input.squeeze()     # seq_len x d_model
        ff_out_full = enc_ff_output.squeeze()   # seq_len x d_model
        ff_in = ff_in_full[:N_rows]
        ff_out = ff_out_full[:N_rows]

        return dict(ff_in = ff_in, ff_out = ff_out)

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


 




