
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


def get_top_N_values(input_vals, N):
    # Get the indices of the N largest elements
    indices = np.argpartition(input_vals, N)[-N:]

    # Return the N largest elements and their indices
    return input_vals[indices], indices

def plot_MI_bars(QK_scored_V_head_mi, word_list):
    QV_scored_MI_estimate = QK_scored_V_head_mi["QV_scored_MI_estimate"]
    KV_scored_MI_estimate = QK_scored_V_head_mi["KV_scored_MI_estimate"]
    input_words = QK_scored_V_head_mi["input_words"]

    # np.fill_diagonal(QK_MI_estimate, 0)

    for word in word_list:
        word_index = input_words.index(word)
        QV_MI = QV_scored_MI_estimate[word_index]
        KV_MI = KV_scored_MI_estimate[word_index]

        fig, axs = plt.subplots(2,1)
        x_vals = np.arange(0, len(input_words))

        # Plot Q-K MI
        top_vals, N_top_indices = get_top_N_values(QV_MI, 3)
        top_indices = np.argwhere(QV_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        axs[0].bar(x_vals, QV_MI, color=bar_colors)
        axs[0].set_title("Q-V MI")
        axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[0].set_ylabel("Mutual Information")

        top_vals, N_top_indices = get_top_N_values(KV_MI, 3)
        top_indices = np.argwhere(KV_MI >= min(top_vals)).squeeze().tolist()
        bar_colors = ['red' if i in top_indices else 'blue' for i in range(len(input_words))]
        axs[1].bar(x_vals, KV_MI, color=bar_colors)
        axs[1].set_title("K-V MI")
        axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
        axs[1].set_ylabel("Mutual Information")

        fig.suptitle(f"Q-V and K-V MI after scoring for word '{word}'")
        plt.subplots_adjust(hspace=0.5)
        plt.show(block=True)

def plot_QK_scored_V_head_mi(QK_scored_V_head_mi, head_id, epoch, attention_layer, sentence_id):
    QV_MI_estimate = QK_scored_V_head_mi["QV_MI_estimate"]
    KV_MI_estimate = QK_scored_V_head_mi["KV_MI_estimate"]
    QV_scored_MI_estimate = QK_scored_V_head_mi["QV_scored_MI_estimate"]
    KV_scored_MI_estimate = QK_scored_V_head_mi["KV_scored_MI_estimate"]
    input_words = QK_scored_V_head_mi["input_words"]

    max_val = np.max([np.max(QV_MI_estimate), np.max(KV_MI_estimate), np.max(QV_scored_MI_estimate), np.max(KV_scored_MI_estimate)])

    fig, axs = plt.subplots(2,2)
    img = axs[0, 0].imshow(QV_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    for i in range(QV_MI_estimate.shape[0]):
        for j in range(QV_MI_estimate.shape[1]):
            text = axs[0, 0].text(j, i, f"{QV_MI_estimate[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title(f"MI between the Q and V heads")
    axs[0, 0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0, 0].set_yticks(range(0, len(input_words)), input_words, rotation=0)

    img = axs[0, 1].imshow(KV_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    for i in range(KV_MI_estimate.shape[0]):
        for j in range(KV_MI_estimate.shape[1]):
            text = axs[0, 1].text(j, i, f"{KV_MI_estimate[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title(f"MI between the K and V heads")
    axs[0, 1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0, 1].set_yticks(range(0, len(input_words)), input_words, rotation=0)

    img = axs[1, 0].imshow(QV_scored_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    for i in range(QV_scored_MI_estimate.shape[0]):
        for j in range(QV_scored_MI_estimate.shape[1]):
            text = axs[1, 0].text(j, i, f"{QV_scored_MI_estimate[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title(f"MI between the Q and V heads")
    axs[1, 0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1, 0].set_yticks(range(0, len(input_words)), input_words, rotation=0)

    img = axs[1, 1].imshow(KV_scored_MI_estimate.T, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    for i in range(KV_scored_MI_estimate.shape[0]):
        for j in range(KV_scored_MI_estimate.shape[1]):
            text = axs[1, 1].text(j, i, f"{KV_scored_MI_estimate[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title(f"MI between the K and V heads")
    axs[1, 1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1, 1].set_yticks(range(0, len(input_words)), input_words, rotation=0)

    fig.colorbar(img, ax=axs.ravel().tolist())
    fig.suptitle(f"Mutual Information between Q and K heads and scored V head\nfor epoch {epoch}, head {head_id}, attention layer {attention_layer}, sentence {sentence_id}")
    plt.show(block=True)


def compute_QK_scored_V_head_mi(analyzer, QKV_head_database, head_id, attention_layer, sentence_tokens):
    QKV_dict = QKV_head_database[f"attention_{attention_layer}"]

    Q_heads = QKV_dict["Q_head"]
    K_heads = QKV_dict["K_head"]
    V_heads = QKV_dict["V_head"]
    scored_V_heads = QKV_dict["scored_V_head"]

    Q_head = Q_heads[head_id]
    K_head = K_heads[head_id]
    V_head = V_heads[head_id]
    scored_V_head = scored_V_heads[head_id]

    # Dimensions of the MI matrix will be N_rows x N_rows
    N_rows = Q_head.shape[0] 
    QV_MI_estimate = np.zeros((N_rows, N_rows))
    KV_MI_estimate = np.zeros((N_rows, N_rows))
    QV_scored_MI_estimate = np.zeros((N_rows, N_rows))
    KV_scored_MI_estimate = np.zeros((N_rows, N_rows))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between Q_head and V_head ...")
    for i, j in tqdm(ij_pos):
        X = Q_head[i]
        Y = V_head[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        QV_MI_estimate[i, j] = MI

    print("\nComputing the mutual information between K_head and V_head ...")
    for i, j in tqdm(ij_pos):
        X = K_head[i]
        Y = V_head[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        KV_MI_estimate[i, j] = MI

    print("\nComputing the mutual information between Q_head and scored V_head ...")
    for i, j in tqdm(ij_pos):
        X = Q_head[i]
        Y = scored_V_head[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        QV_scored_MI_estimate[i, j] = MI

    print("\nComputing the mutual information between K_head and scored V_head ...")
    for i, j in tqdm(ij_pos):
        X = K_head[i]
        Y = scored_V_head[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        KV_scored_MI_estimate[i, j] = MI

    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_tgt_word_from_token(token))

    QKV_scored_head_mi = \
        dict(QV_MI_estimate=QV_MI_estimate, 
             KV_MI_estimate=KV_MI_estimate, 
             QV_scored_MI_estimate=QV_scored_MI_estimate, 
             KV_scored_MI_estimate=KV_scored_MI_estimate,
             input_words=input_words)

    return QKV_scored_head_mi

# Main function of this script
def main():
    print("Computing the mutual information between the Q, K attention heads and the scored V head ...")

    epoch = 19
    attention_layer = 2
    sentence_id = 3
    decoder_token_id = 10
    head_id = 2

    save_file = Path(f"data/cross_QK_scored_V_head_mi_epoch_{epoch}_head_{head_id}_layer_{attention_layer}_sentence_{sentence_id}.pt")

    if save_file.exists():
        print(f"QK and scored V head mutual information file {str(save_file)} found. Loading it ...")
        QK_scored_V_head_mi = torch.load(save_file, weights_only=False)
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
        QK_scored_V_head_mi = compute_QK_scored_V_head_mi(analyzer, QKV_head_database, head_id, attention_layer, tgt_sentence_tokens)

        # Save the file
        torch.save(QK_scored_V_head_mi, save_file)
    
    plot_QK_scored_V_head_mi(QK_scored_V_head_mi, head_id, epoch, attention_layer, sentence_id)

    word_list = ["chien", "dans", "wagon"]
    plot_MI_bars(QK_scored_V_head_mi, word_list)

# Entry point of the script
if __name__ == '__main__':
    main()