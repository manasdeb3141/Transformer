
import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


# Classes implemented by this application
from ModelConfig import LangModelConfig
from TransformerAnalyzer import TransformerAnalyzer
from mutual_info_estimator import MutualInfoEstimator

# Functions implemented by this application
from get_sentence_tokens import get_sentence_tokens
from get_QKV import get_cross_atten_QKV_head
from get_attention_scores import get_cross_attention_scores



def get_top_N_values(input_vals, N):
    # Get the indices of the N largest elements
    indices = np.argpartition(input_vals, N)[-N:]

    # Return the N largest elements and their indices
    return input_vals[indices], indices

def plot_MI_bars(QK_head_mi_dict, word_list):
    QK_head_mi = QK_head_mi_dict["QK_head_mi"]
    attention_scores = QK_head_mi_dict["attention_scores"]

    QK_MI_estimate = QK_head_mi["QK_MI_estimate"]
    input_words = QK_head_mi["input_words"]

    # np.fill_diagonal(QK_MI_estimate, 0)

    for word in word_list:
        word_index = input_words.index(word)
        QK_MI = QK_MI_estimate[word_index]
        attention_score = attention_scores[word_index]

        fig, axs = plt.subplots(2,1)
        x_vals = np.arange(0, len(input_words))

        # Plot Q-K MI
        top_vals, N_top_indices = get_top_N_values(QK_MI, 3)
        top_indices = np.argwhere(QK_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        axs[0].bar(x_vals, QK_MI, color=bar_colors)
        axs[0].set_title("Q-K MI")
        axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[0].set_ylabel("Mutual Information")
        axs[0].grid(True)

        # Plot attention score
        top_vals, N_top_indices = get_top_N_values(attention_score, 3)
        top_indices = np.argwhere(attention_score >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        axs[1].bar(x_vals, attention_score, color=bar_colors)
        axs[1].set_title("Attention Score")
        axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[1].set_ylabel("Score")
        axs[1].grid(True)

        fig.suptitle(f"Comparison of Q-K MI and Attention Score for word '{word}'")
        plt.subplots_adjust(hspace=0.5)
        plt.show(block=True)

def plot_QK_head_mi_attn_scores(QK_head_mi_dict, head_id, epoch, attention_layer, sentence_id):
    QK_head_mi = QK_head_mi_dict["QK_head_mi"]
    attention_scores = QK_head_mi_dict["attention_scores"]

    QK_MI_estimate = QK_head_mi["QK_MI_estimate"]
    input_words = QK_head_mi["input_words"]

    fig, axs = plt.subplots(1,2)
    img = axs[0].imshow(QK_MI_estimate.T, cmap=plt.cm.Wistia)
    for i in range(QK_MI_estimate.shape[0]):
        for j in range(QK_MI_estimate.shape[1]):
            text = axs[0].text(j, i, f"{QK_MI_estimate[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between the Q and K heads")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(attention_scores.T, cmap=plt.cm.Wistia)
    for i in range(attention_scores.shape[0]):
        for j in range(attention_scores.shape[1]):
            text = axs[1].text(j, i, f"{attention_scores[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[1].set_title(f"Attention Scores")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    fig.colorbar(img, ax=axs[1])

    fig.suptitle(f"Mutual Information between Q and K heads and Attention Scores\nfor epoch {epoch}, head {head_id}, attention layer {attention_layer}, sentence {sentence_id}")
    plt.show(block=True)


def compute_QK_head_mi(analyzer, QKV_head_database, head_id, attention_layer, sentence_tokens):
    QKV_dict = QKV_head_database[f"attention_{attention_layer}"]

    Q_heads = QKV_dict["Q_head"]
    K_heads = QKV_dict["K_head"]

    Q_head = Q_heads[head_id]
    K_head = K_heads[head_id]

    # Dimensions of the MI matrix will be N_rows x N_rows
    N_rows = Q_head.shape[0] 
    QK_MI_estimate = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between the rows of Q_head and K_head ...")
    for i, j in tqdm(ij_pos):
        X = Q_head[i]
        Y = K_head[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        QK_MI_estimate[i, j] = MI

    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_tgt_word_from_token(token))

    QKV_head_mi = \
        dict(QK_MI_estimate=QK_MI_estimate, input_words=input_words)

    return QKV_head_mi

# Main function of this script
def main():
    print("Computing the mutual information between the Q, K attention heads and comparing with attention scores ...")

    epoch = 19
    attention_layer = 2
    sentence_id = 3
    decoder_token_id = 10
    head_id = 2

    save_file = Path(f"data/cross_QKV_head_mi_attn_scores_epoch_{epoch}_layer_{attention_layer}_head_{head_id}_sentence_{sentence_id}.pt")

    if save_file.exists():
        print(f"QKV head mutual information file {str(save_file)} found. Loading it ...")
        QK_head_mi_dict = torch.load(save_file, weights_only=False)
    else:
        cfg_obj = LangModelConfig()
        model_config = cfg_obj.get_config()

        model_config["tokenizer_dir"] = "../../../model_data/opus_books_en_fr/tokens"
        model_config["model_dir"] = "../../../model_data/opus_books_en_fr/weights"
        model_config["analyze_dir"] = "../../../model_data/opus_books_en_fr/probes_8"
        #model_config["tokenizer_dir"] = "../../../model_data_d32/opus_books_en_fr/tokens"
        #model_config["model_dir"] = "../../../model_data_d32/opus_books_en_fr/weights"

        # Dictionary of probe file names
        probe_config = cfg_obj.get_probes()

        # Instantiate the Transformer's analyzer object.
        # Load the tokenizer and instantiate the ProbeManager objects
        analyzer = TransformerAnalyzer(model_config, probe_config)
        analyzer.run()

        # For this epoch, load all the decoder layer probe files from disk
        analyzer.load_decoder_probes(epoch)
        
        # Number of input sentences in this epoch
        N_inputs = len(analyzer.decoder_probe._probe_in)

        # Get the tokens of the source and target sentences
        _, _, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id, decoder_token_id)

        # Get the query, key, value arrays for all the attention layers of this input sentence
        QKV_head_database = get_cross_atten_QKV_head(analyzer, sentence_id, decoder_token_id)
        attention_scores = get_cross_attention_scores(analyzer, sentence_id, head_id, decoder_token_id, attention_layer)
        QK_head_mi = compute_QK_head_mi(analyzer, QKV_head_database, head_id, attention_layer, tgt_sentence_tokens)

        # Save the file
        QK_head_mi_dict = dict(QK_head_mi=QK_head_mi, attention_scores=attention_scores)
        torch.save(QK_head_mi_dict, save_file)
    
    plot_QK_head_mi_attn_scores(QK_head_mi_dict, head_id, epoch, attention_layer, sentence_id)

    # word_list = ["dog", "car", "cat", "van"]
    word_list = ["chien", "dans", "wagon"]
    plot_MI_bars(QK_head_mi_dict, word_list)

# Entry point of the script
if __name__ == '__main__':
    main()