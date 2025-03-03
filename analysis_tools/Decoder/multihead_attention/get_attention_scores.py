
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_self_attention_scores(analyzer : TransformerAnalyzer, sentence_id : int, head_id : int, decoder_token_id: int, attention_layer : int):
    match attention_layer:
        case 0:
            # Attention layer 0
            dec_attn_input = analyzer.dec_0_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 1:
            # Attention layer 1
            dec_attn_input = analyzer.dec_1_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 2:
            # Attention layer 2
            dec_attn_input = analyzer.dec_2_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 3:
            # Attention layer 3
            dec_attn_input = analyzer.dec_3_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 4:
            # Attention layer 4
            dec_attn_input = analyzer.dec_4_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 5:
            # Attention layer 5
            dec_attn_input = analyzer.dec_5_attn_probe._probe_in[sentence_id][decoder_token_id]
    
    attention_scores = dec_attn_input["attention_scores"].squeeze()
    return attention_scores[head_id][:decoder_token_id+1, :decoder_token_id+1]



def get_cross_attention_scores(analyzer : TransformerAnalyzer, sentence_id : int, head_id: int, decoder_token_id : int, attention_layer : int):
    match attention_layer:
        case 0:
            # Attention layer 0
            dec_attn_input = analyzer.dec_0_cross_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 1:
            # Attention layer 1
            dec_attn_input = analyzer.dec_1_cross_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 2:
            # Attention layer 2
            dec_attn_input = analyzer.dec_2_cross_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 3:
            # Attention layer 3
            dec_attn_input = analyzer.dec_3_cross_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 4:
            # Attention layer 4
            dec_attn_input = analyzer.dec_4_cross_attn_probe._probe_in[sentence_id][decoder_token_id]

        case 5:
            # Attention layer 5
            dec_attn_input = analyzer.dec_5_cross_attn_probe._probe_in[sentence_id][decoder_token_id]

    attention_scores = dec_attn_input["attention_scores"].squeeze()
    return attention_scores[head_id][:decoder_token_id+1, :decoder_token_id+1]



