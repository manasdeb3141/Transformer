
import sys
sys.path.append('../..')
sys.path.append('../utils')

import numpy as np

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer
    

def get_attention_scores(analyzer : TransformerAnalyzer, sentence_id, N_src_tokens, attention_layer) -> np.array:
    match attention_layer:
        case 0:
            enc_attn_input = analyzer.enc_0_attn_probe._probe_in[sentence_id]

        case 1:
            enc_attn_input = analyzer.enc_1_attn_probe._probe_in[sentence_id]

        case 2:
            enc_attn_input = analyzer.enc_2_attn_probe._probe_in[sentence_id]

        case 3:
            enc_attn_input = analyzer.enc_3_attn_probe._probe_in[sentence_id]

        case 4:
            enc_attn_input = analyzer.enc_4_attn_probe._probe_in[sentence_id]

        case 5:
            enc_attn_input = analyzer.enc_5_attn_probe._probe_in[sentence_id]

        case _:
            print(f"Invalid attention layer {attention_layer}")
            return

    attention_scores = enc_attn_input["attention_scores"].squeeze()
    return attention_scores[:, :N_src_tokens, :N_src_tokens]
