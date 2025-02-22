import sys
sys.path.append('../..')
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt

# Classes implemented by this application
from TransformerAnalyzer import TransformerAnalyzer

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_attention_scores import get_attention_scores

def plot_attention_scores(attention_scores, input_words, plot_title) -> None:
    N_heads = attention_scores.shape[0]

    fig, axs = plt.subplots(2, 4)
    for head in range(N_heads):
        a = head//4
        b = head%4
        attention_scores_matrix = attention_scores[head]
        # im = axs[a, b].imshow(attention_scores_matrix, cmap=plt.cm.Wistia, interpolation='none', origin='lower', extent=[0, attention_scores.shape[1]-1, 0, attention_scores.shape[2]-1])
        im = axs[a, b].imshow(attention_scores_matrix, cmap=plt.cm.Wistia, interpolation='none', origin='lower')

        # Add text annotations
        for i in range(attention_scores_matrix.shape[0]):
            for j in range(attention_scores_matrix.shape[1]):
                text = axs[a, b].text(j, i, f"{attention_scores_matrix[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)

        axs[a, b].set_xticks(range(0, len(input_words)), input_words, rotation=45)
        axs[a, b].set_yticks(range(0, len(input_words)), input_words, rotation=45)
        axs[a, b].set_aspect('auto')
        axs[a, b].set_title(f"Head {head}")

    fig.suptitle(plot_title)

    plt.subplots_adjust(hspace=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Get the current figure manager to maximize the plot window
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state('zoomed')

    plt.show(block=False)


def process_attention_scores(analyzer : TransformerAnalyzer):
    print("Computing the attention scores ...")
    # epochs_to_analyze = np.arange(0, 20, 1)
    # epochs_to_analyze = [0, 4, 9, 14, 19]

    # Input sentence ID
    sentence_id = 0
    epoch = 19
    attention_layer = 5

    # For this epoch, load all the encoder layer probe files from disk
    analyzer.load_encoder_probes(epoch)

    # Get the number of tokens of the sentence
    N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)

    # Get the input sentence words corresponding to the tokens
    input_words = list()
    for token in src_sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    # Get the attention scores for all the heads of the specified attention layer
    attention_scores = get_attention_scores(analyzer, sentence_id, N_src_tokens, attention_layer)

    # Plot the attention scores
    plot_title = f"Attention Scores of encoder layer {attention_layer}"
    plot_attention_scores(attention_scores, input_words, plot_title)