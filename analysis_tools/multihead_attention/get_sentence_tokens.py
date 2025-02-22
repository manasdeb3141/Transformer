
import sys
sys.path.append('../..')
sys.path.append('../utils')

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_sentence_tokens(analyzer : TransformerAnalyzer, sentence_id : int) -> Tuple[int, np.array, int, np.array]:
    # Get the number of tokens in this source sentence
    # from the encoder block's input mask probe
    src_mask=analyzer.encoder_probe._probe_in[sentence_id]["mask"]
    N_src_tokens = np.count_nonzero(np.squeeze(src_mask))

    # Get the encoder's embedding layer input and output probes
    enc_embedding_input = analyzer.enc_embedding_probe._probe_in[sentence_id]
    enc_embedding_output = analyzer.enc_embedding_probe._probe_out[sentence_id]

    # From the encoder's embedding layer input probe get the source and target tokens
    src_tokens = enc_embedding_input["src_tokens"]
    src_sentence_tokens = np.squeeze(src_tokens)[:N_src_tokens]

    tgt_mask=enc_embedding_input["tgt_mask"]
    N_tgt_tokens = np.count_nonzero(np.squeeze(tgt_mask))
    tgt_tokens = enc_embedding_input["tgt_tokens"]
    tgt_sentence_tokens = np.squeeze(tgt_tokens)[:N_tgt_tokens]

    return N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens