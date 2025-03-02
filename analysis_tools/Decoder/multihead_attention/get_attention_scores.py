
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

def get_self_attention_scores(analyzer : TransformerAnalyzer, sentence_id : int, decoder_token_id: int):
    # Attention layer 0
    dec_attn_input = analyzer.dec_0_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_0 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 1
    dec_attn_input = analyzer.dec_1_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_1 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 2
    dec_attn_input = analyzer.dec_2_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_2 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 3
    dec_attn_input = analyzer.dec_3_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_3 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 4
    dec_attn_input = analyzer.dec_4_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_4 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 5
    dec_attn_input = analyzer.dec_5_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_5 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    atten_scores_dict = { 'attention_0': atten_scores_0,
                          'attention_1': atten_scores_1,
                          'attention_2': atten_scores_2,
                          'attention_3': atten_scores_3,
                          'attention_4': atten_scores_4,
                          'attention_5': atten_scores_5}


    return atten_scores_dict 


def get_cross_atten_scores(analyzer : TransformerAnalyzer, sentence_id : int, decoder_token_id : int):
    # Attention layer 0
    dec_attn_input = analyzer.dec_0_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_0 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 1
    dec_attn_input = analyzer.dec_1_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_1 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 2
    dec_attn_input = analyzer.dec_2_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_2 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 3
    dec_attn_input = analyzer.dec_3_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_3 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 4
    dec_attn_input = analyzer.dec_4_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_4 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    # Attention layer 5
    dec_attn_input = analyzer.dec_5_cross_attn_probe._probe_in[sentence_id][decoder_token_id]
    atten_scores = dec_attn_input["attention_scores"].squeeze()
    atten_scores_5 = atten_scores[:, :decoder_token_id+1, :decoder_token_id+1]

    atten_scores_dict = { 'attention_0': atten_scores_0,
                          'attention_1': atten_scores_1,
                          'attention_2': atten_scores_2,
                          'attention_3': atten_scores_3,
                          'attention_4': atten_scores_4,
                          'attention_5': atten_scores_5}


    return atten_scores_dict 


