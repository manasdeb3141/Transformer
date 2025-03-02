import sys
sys.path.append('../../..')
sys.path.append('../../utils')

import numpy as np
import matplotlib.pyplot as plt

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_QKV import get_query_key_value_head
from matrix_mutual_information import compute_matrix_mi
from get_attention_scores import get_attention_scores


def plot_mi_attention_scores(MI_mat, input_words, attention_scores, plot_title) -> None:
    fig, axs = plt.subplots(1, 2)

    # im0 = axs[0].imshow(MI, cmap=plt.cm.jet, origin='lower')
    # im0 = axs[0].imshow(MI_mat, cmap=plt.cm.jet)
    im0 = axs[0].imshow(MI_mat, cmap=plt.cm.Wistia)
    for i in range(MI_mat.shape[0]):
        for j in range(MI_mat.shape[1]):
            text = axs[0].text(j, i, f"{MI_mat[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    axs[0].set_aspect('auto')
    axs[0].set_title(f"Mutual Information between Q and K heads")
    fig.colorbar(im0, ax=axs[0])

    # Plot the attention scores
    # im1 = axs[1].imshow(attention_scores, cmap=plt.cm.Wistia, interpolation='none', origin='lower')
    im1 = axs[1].imshow(attention_scores, cmap=plt.cm.Wistia)
    for i in range(attention_scores.shape[0]):
        for j in range(attention_scores.shape[1]):
            text = axs[1].text(j, i, f"{attention_scores[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=45)
    axs[1].set_aspect('auto')
    axs[1].set_title(f"Attention scores")
    fig.colorbar(im1, ax=axs[1])

    fig.suptitle(plot_title)

    # Get the current figure manager to maximize the plot window
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state('zoomed')

    plt.savefig("mi_attention_scores.png")
    plt.show(block=False)



def process_mi_attention_scores(analyzer : TransformerAnalyzer):
    head = 0
    sentence_id = 3
    epoch = 19
    attention_layer = 0

    print("Computing the mutual information and attention scores with the following parameters:")
    print(f"Head: {head}, Sentence ID: {sentence_id}, Epoch: {epoch}, Attention Layer: {attention_layer}")

    analyzer.load_encoder_probes(epoch)
    N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)
    QKV_dict = get_query_key_value_head(analyzer, sentence_id, N_src_tokens)

    query = QKV_dict[f'attention_{attention_layer}']["query"]
    key = QKV_dict[f'attention_{attention_layer}']["key"]

    seq_len = query.shape[1]
    query_head = query[head]
    key_head = key[head]

    MI_dict = compute_matrix_mi(query_head, key_head, N_src_tokens)
    MI_mat = MI_dict["MI"]

    # Get the input sentence words corresponding to the tokens
    input_words = list()
    for token in src_sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    attention_scores = get_attention_scores(analyzer, sentence_id, N_src_tokens, attention_layer)

    plot_title = f"Encoder attention layer {attention_layer}, head {head}, epoch {epoch}"
    plot_mi_attention_scores(MI_mat, input_words, attention_scores[attention_layer], plot_title)
