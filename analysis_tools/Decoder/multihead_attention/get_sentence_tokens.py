
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_sentence_tokens(analyzer : TransformerAnalyzer, sentence_id : int, token_id : int) -> Tuple[int, int, np.array]:
    # Get the number of tokens in this source sentence
    # from the encoder block's input mask probe
    src_mask=analyzer.decoder_probe._probe_in[sentence_id][token_id]["src_mask"]
    N_src_tokens = np.count_nonzero(np.squeeze(src_mask))

    # Get the number of tokens in the sentence input to the decoder
    tgt_mask=analyzer.decoder_probe._probe_in[sentence_id][token_id]["tgt_mask"]
    N_tgt_tokens = tgt_mask.shape[1]

    # Get the decoder's embedding layer input and output probes
    dec_embedding_input = analyzer.dec_embedding_probe._probe_in[sentence_id][token_id]

    # From the decoder's embedding layer input probe get the target tokens
    tgt_sentence_tokens = np.squeeze(dec_embedding_input)[:N_tgt_tokens]
    # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

    # From the decoder's embedding layer input probe get the target tokens
    tgt_sentence_tokens = np.squeeze(dec_embedding_input)[:N_tgt_tokens]
    # print(f"Target sentence: {self._tokenizer_tgt.decode(tgt_sentence_tokens)}")

    return N_src_tokens, N_tgt_tokens, tgt_sentence_tokens