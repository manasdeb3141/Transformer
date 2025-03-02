
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
from get_QKV import get_query_key_value_matrix

def plot_mi_image(Q_MI_estimate, K_MI_estimate, V_MI_estimate, input_words, epoch, attention_layer, sentence_id):
    fig, axs = plt.subplots(1,3)
    img = axs[0].imshow(Q_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # img = ax.imshow(query_MI_estimate.T, cmap=plt.cm.jet)
    # axs[0].set_aspect('auto')
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between the rows of Q and Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(K_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # axs[1].set_aspect('auto')
    axs[0].set_aspect('equal')
    axs[1].set_title(f"MI between the rows of K and K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(V_MI_estimate.T, vmin=0, vmax=0.1, cmap=plt.cm.Wistia)
    # axs[2].set_aspect('auto')
    axs[0].set_aspect('equal')
    axs[2].set_title(f"MI between the rows of V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[2])

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(img, cax=cbar_ax)
    fig.colorbar(img, ax=axs.ravel().tolist())

    fig.suptitle(f"Mutual Information between the the rows of Q, K, V and \nQ', K', V' for epoch {epoch}, attention layer {attention_layer}, sentence {sentence_id}")
    plt.show(block=True)

def plot_QKV_mi(QKV_entropy_mi, epoch, attention_layer, sentence_id):
    Q_MI_estimate=QKV_entropy_mi["Q_MI_estimate"]
    K_MI_estimate=QKV_entropy_mi["K_MI_estimate"]
    V_MI_estimate=QKV_entropy_mi["V_MI_estimate"]
    input_words = QKV_entropy_mi["input_words"]

    plot_mi_image(Q_MI_estimate, K_MI_estimate, V_MI_estimate, input_words, epoch, attention_layer, sentence_id)

    # Plot the MI after zeroing out the diagonal elements
    np.fill_diagonal(Q_MI_estimate, 0)
    np.fill_diagonal(K_MI_estimate, 0)
    np.fill_diagonal(V_MI_estimate, 0)

    plot_mi_image(Q_MI_estimate, K_MI_estimate, V_MI_estimate, input_words, epoch, attention_layer, sentence_id)


def plot_QKV_entropy(QKV_entropy_mi, epoch, attention_layer, sentence_id):
    Q_joint_entropy_estimate=QKV_entropy_mi["Q_joint_entropy_estimate"]
    K_joint_entropy_estimate=QKV_entropy_mi["K_joint_entropy_estimate"]
    V_joint_entropy_estimate=QKV_entropy_mi["V_joint_entropy_estimate"] 
    Q_entropy_estimate = QKV_entropy_mi["Q_entropy_estimate"]
    K_entropy_estimate = QKV_entropy_mi["K_entropy_estimate"]
    V_entropy_estimate = QKV_entropy_mi["V_entropy_estimate"]
    Q_prime_entropy_estimate = QKV_entropy_mi["Q_prime_entropy_estimate"]
    K_prime_entropy_estimate = QKV_entropy_mi["K_prime_entropy_estimate"]
    V_prime_entropy_estimate = QKV_entropy_mi["V_prime_entropy_estimate"]
    input_words = QKV_entropy_mi["input_words"]

    max_val = np.max([np.max(Q_joint_entropy_estimate), np.max(K_joint_entropy_estimate), np.max(V_joint_entropy_estimate)])

    fig, axs = plt.subplots(1,3)
    img = axs[0].imshow(Q_joint_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Joint entropy between\nthe rows of Q and Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[0])

    img = axs[1].imshow(K_joint_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Joint entropy between\nthe rows of K and K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[1])

    img = axs[2].imshow(V_joint_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Joint_entropy between\nthe rows of V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    # fig.colorbar(img, ax=axs[2])

    fig.suptitle(f"Joint entropy between the the rows of Q, K, V and \nQ', K', V' for epoch {epoch}, attention layer {attention_layer}, sentence {sentence_id}")
    fig.colorbar(img, ax=axs.ravel().tolist())
    plt.show(block=True)


    # Plot the entropies of Q, K, V, and Q', K', V'
    max_val = np.max([np.max(Q_entropy_estimate), np.max(K_entropy_estimate), np.max(V_entropy_estimate), np.max(Q_prime_entropy_estimate), np.max(K_prime_entropy_estimate), np.max(V_prime_entropy_estimate)])
    fig, axs = plt.subplots(2,3)
    img = axs[0, 0].imshow(Q_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.jet)
    axs[0, 0].set_aspect('auto')
    axs[0, 0].set_xticks([0])
    axs[0, 0].set_title(f"Q")
    # fig.colorbar(img, ax=axs[0, 0])
    img = axs[0, 1].imshow(K_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.jet)
    axs[0, 1].set_aspect('auto')
    axs[0, 1].set_xticks([0])
    axs[0, 1].set_title(f"K")
    # fig.colorbar(img, ax=axs[0, 1])
    img = axs[0, 2].imshow(V_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.jet)
    axs[0, 2].set_aspect('auto')
    axs[0, 2].set_xticks([0])
    axs[0, 2].set_title(f"V")
    # fig.colorbar(img, ax=axs[0, 2])

    img = axs[1, 0].imshow(Q_prime_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.jet)
    axs[1, 0].set_aspect('auto')
    axs[1, 0].set_xticks([0])
    axs[1, 0].set_title(f"Q'")
    # fig.colorbar(img, ax=axs[1, 0])
    img = axs[1, 1].imshow(K_prime_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.jet)
    axs[1, 1].set_aspect('auto')
    axs[1, 1].set_xticks([0])
    axs[1, 1].set_title(f"K'")
    # fig.colorbar(img, ax=axs[1, 1])
    img = axs[1, 2].imshow(V_prime_entropy_estimate, vmin=0, vmax=max_val, cmap=plt.cm.jet)
    axs[1, 2].set_aspect('auto')
    axs[1, 2].set_xticks([0])
    axs[1, 2].set_title(f"V'")
    # fig.colorbar(img, ax=axs[1, 2])

    fig.suptitle(f"Entropy of Q, K, V and \nQ', K', V' for epoch {epoch}, attention layer {attention_layer}, sentence {sentence_id}")
    fig.colorbar(img, ax=axs.ravel().tolist())
    plt.show(block=True)


def compute_QKV_entropy_mi(analyzer, QKV_database, attention_layer, sentence_tokens):
    QKV_dict = QKV_database[f"attention_{attention_layer}"]

    Q = QKV_dict["x"]
    K = QKV_dict["x"]
    V = QKV_dict["x"]
    Q_prime = QKV_dict["query"]
    K_prime = QKV_dict["key"]
    V_prime = QKV_dict["value"]

    # Dimensions of the MI and joint entropy matrix will be N_rows x N_rows
    N_rows = Q.shape[0] 
    Q_MI_estimate = np.zeros((N_rows, N_rows))
    K_MI_estimate = np.zeros((N_rows, N_rows))
    V_MI_estimate = np.zeros((N_rows, N_rows))
    Q_joint_entropy_estimate = np.zeros((N_rows, N_rows))
    K_joint_entropy_estimate = np.zeros((N_rows, N_rows))
    V_joint_entropy_estimate = np.zeros((N_rows, N_rows))
    Q_entropy_estimate = np.zeros((N_rows, 1))
    K_entropy_estimate = np.zeros((N_rows, 1))
    V_entropy_estimate = np.zeros((N_rows, 1))
    Q_prime_entropy_estimate = np.zeros((N_rows, 1))
    K_prime_entropy_estimate = np.zeros((N_rows, 1))
    V_prime_entropy_estimate = np.zeros((N_rows, 1))

    i = np.arange(0, N_rows, 1)
    j = np.arange(0, N_rows, 1)
    i_pos, j_pos = np.meshgrid(i, j)
    ij_pos = np.vstack([i_pos.ravel(), j_pos.ravel()]).T

    print("\nComputing the mutual information between the rows of Q and Q_prime ...")
    for i, j in tqdm(ij_pos):
        X = Q[i]
        Y = Q_prime[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        Q_MI_estimate[i, j] = MI
        Q_joint_entropy_estimate[i, j] = MI_data["H_XY"]
        Q_entropy_estimate[i] = MI_data["H_X"]
        Q_prime_entropy_estimate[j] = MI_data["H_Y"]

    print("\nComputing the mutual information between the rows of K and K_prime ...")
    for i, j in tqdm(ij_pos):
        X = K[i]
        Y = K_prime[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        K_MI_estimate[i, j] = MI
        K_joint_entropy_estimate[i, j] = MI_data["H_XY"]
        K_entropy_estimate[i] = MI_data["H_X"]
        K_prime_entropy_estimate[j] = MI_data["H_Y"]

    print("\nComputing the mutual information between the rows of V and V_prime ...")
    for i, j in tqdm(ij_pos):
        X = V[i]
        Y = V_prime[j]
        MI_estimator = MutualInfoEstimator(X, Y)
        _, MI_data = MI_estimator.kernel_MI(KDE_module='sklearn')
        MI = MI_data["MI"]
        V_MI_estimate[i, j] = MI
        V_joint_entropy_estimate[i, j] = MI_data["H_XY"]
        V_entropy_estimate[i] = MI_data["H_X"]
        V_prime_entropy_estimate[j] = MI_data["H_Y"]

    input_words = list()
    for token in sentence_tokens:
        input_words.append(analyzer.get_src_word_from_token(token))

    QKV_entropy_mi = \
        dict(Q_MI_estimate=Q_MI_estimate, K_MI_estimate=K_MI_estimate, V_MI_estimate=V_MI_estimate,
                Q_joint_entropy_estimate=Q_joint_entropy_estimate, K_joint_entropy_estimate=K_joint_entropy_estimate, V_joint_entropy_estimate=V_joint_entropy_estimate,
                Q_entropy_estimate=Q_entropy_estimate, K_entropy_estimate=K_entropy_estimate, V_entropy_estimate=V_entropy_estimate,
                Q_prime_entropy_estimate=Q_prime_entropy_estimate, K_prime_entropy_estimate=K_prime_entropy_estimate, V_prime_entropy_estimate=V_prime_entropy_estimate,
                input_words=input_words)

    return QKV_entropy_mi

# Main function of this script
def main():
    print("Computing the entropy and mutual information of between Q, K, V and Q', K', V' of the encoder ...")

    epoch = 19
    attention_layer = 2
    sentence_id = 3

    save_file = Path(f"data/QKV_entropy_mi_epoch_{epoch}_layer_{attention_layer}_sentence_{sentence_id}.pt")

    if save_file.exists():
        print(f"QKV entropy and mutual information file {str(save_file)} found. Loading it ...")
        QKV_entropy_mi = torch.load(save_file, weights_only=False)
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

        # For this epoch, load all the encoder layer probe files from disk
        analyzer.load_encoder_probes(epoch)
        
        # Number of input sentences in this epoch
        N_inputs = len(analyzer.encoder_probe._probe_in)

        # Get the tokens of the source and target sentences
        N_src_tokens, src_sentence_tokens, N_tgt_tokens, tgt_sentence_tokens = get_sentence_tokens(analyzer, sentence_id)

        # Get the query, key, value arrays for all the attention layers of this input sentence
        QKV_database = get_query_key_value_matrix(analyzer, sentence_id, N_src_tokens)
        QKV_entropy_mi = compute_QKV_entropy_mi(analyzer, QKV_database, attention_layer, src_sentence_tokens)

        # Save the file
        torch.save(QKV_entropy_mi, save_file)
    
    plot_QKV_mi(QKV_entropy_mi, epoch, attention_layer, sentence_id)
    plot_QKV_entropy(QKV_entropy_mi, epoch, attention_layer, sentence_id)

# Entry point of the script
if __name__ == '__main__':
    main()