
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import seaborn as sns
from matplotlib import colors
from adjustText import adjust_text


# Class implemented by this application
from probability_distances import compute_bhattacharya_coefficient
from probability_distances import compute_jensen_shannon_divergence
from compute_MI import KDE_mutual_info_WSD


# ---------------------------------- Plotting routines start --------------------------------

def plot_QKV_head_MI_WSD_atten_scores(MI_WSD_dict: dict, atten_scores: np.ndarray, input_words: list, epoch: int, enc_layer: int, head_id: int, sentence_id: int):
    # Extract the mutual information matrices
    MI_QK_head = MI_WSD_dict["MI_QK_head"]
    MI_QV_scored_head = MI_WSD_dict["MI_QV_scored_head"]
    MI_KV_scored_head = MI_WSD_dict["MI_KV_scored_head"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Create the figure and axes for plotting the Q-K MI and attention scores
    fig, axs = plt.subplots(1,2)

    # Plot the Q and K heads mutual information matrix
    img = axs[0].imshow(MI_QK_head, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_QK_head.shape[0]):
            for j in range(MI_QK_head.shape[1]):
                text = axs[0].text(j, i, f"{MI_QK_head[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title("MI between Q and K atten heads")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the Q-K attention scores
    img = axs[1].imshow(atten_scores, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(atten_scores.shape[0]):
            for j in range(atten_scores.shape[1]):
                text = axs[1].text(j, i, f"{atten_scores[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title("Attention scores")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Comparison of MI and Attention scores\nfor epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)


    # Create the figure and axes for plotting the Q-V scored and K-V scored MI
    fig, axs = plt.subplots(1,2)

    # Plot the Q-V scored mutual information matrix
    img = axs[0].imshow(MI_QV_scored_head, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_QV_scored_head.shape[0]):
            for j in range(MI_QV_scored_head.shape[1]):
                text = axs[0].text(j, i, f"{MI_QV_scored_head[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between Q and V scored attention heads")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K-V scored mutual information matrix
    img = axs[1].imshow(MI_KV_scored_head, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_KV_scored_head.shape[0]):
            for j in range(MI_KV_scored_head.shape[1]):
                text = axs[1].text(j, i, f"{MI_KV_scored_head[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"MI between K and V scored attention heads")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Mutual Information between the Q, K, and V scored heads\nfor epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Wasserstein distance between the rows of Q and K heads, Q and scored V heads, 
    # K and scored V heads
    # ---------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1,3)

    WSD_QK_head=MI_WSD_dict["WSD_QK_head"]
    WSD_QV_scored_head=MI_WSD_dict["WSD_QV_scored_head"]
    WSD_KV_scored_head=MI_WSD_dict["WSD_KV_scored_head"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q-K Wasserstein distance matrix
    img = axs[0].imshow(WSD_QK_head, cmap=plt.cm.jet)
    if display_values:
        for i in range(WSD_QK_head.shape[0]):
            for j in range(WSD_QK_head.shape[1]):
                text = axs[0].text(j, i, f"{WSD_QK_head[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Wasserstein distance between the Q and K heads")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the Q-V_scored Wasserstein distance matrix
    img = axs[1].imshow(WSD_QV_scored_head, cmap=plt.cm.jet)
    if display_values:
        for i in range(WSD_QV_scored_head.shape[0]):
            for j in range(WSD_QV_scored_head.shape[1]):
                text = axs[1].text(j, i, f"{WSD_QV_scored_head[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Wasserstein distance between the Q and V_scored heads")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K-V_cored Wasserstein distance matrix
    img = axs[2].imshow(WSD_KV_scored_head, cmap=plt.cm.jet)
    if display_values:
        for i in range(WSD_KV_scored_head.shape[0]):
            for j in range(WSD_KV_scored_head.shape[1]):
                text = axs[2].text(j, i, f"{WSD_KV_scored_head[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Wasserstein distance between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Wasserstein distance between the Q-K, Q-V_scored, K-V_scored\nfor epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)


def plot_QKV_head_pdf(QKV_dict: dict, MI_WSD_dict: dict, input_words: list, epoch: int, enc_layer: int, head_id: int, sentence_id: int) -> None:
    P_QK_head = MI_WSD_dict["P_QK_head"]
    P_QV_scored_head = MI_WSD_dict["P_QV_scored_head"]
    P_KV_scored_head = MI_WSD_dict["P_KV_scored_head"]

    # Create the figure and axes
    sns.set_theme(style="darkgrid", rc={'axes.facecolor':'palegoldenrod', 'figure.facecolor':'white'})

    # Word indices to plot
    word_indices=[(4, 7), (4, 10), (10, 13), (7, 13)]

    fig, axs = plt.subplots(4,3)

    # ---------------------------------------------------------------------------------------------
    # Plot the probabilities of Q', K', V' for specific words
    # ---------------------------------------------------------------------------------------------
    for i in range(4):
        row_tuple = word_indices[i]
        X_word = input_words[row_tuple[0]]
        Y_word = input_words[row_tuple[1]]

        prob_data = P_QK_head[row_tuple[0]][row_tuple[1]]["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["xpos"]

        sns.lineplot(x=xpos, y=P_X, label='P(Q)', ax=axs[i, 0])
        sns.lineplot(x=ypos, y=P_Y, label='P(K)', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"P(Q), P(K) for '{X_word}' and '{Y_word}'")

        prob_data = P_QV_scored_head[row_tuple[0]][row_tuple[1]]["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["xpos"]

        sns.lineplot(x=xpos, y=P_X, label='P(Q)', ax=axs[i, 1])
        sns.lineplot(x=ypos, y=P_Y, label='P(V_scored)', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"P(Q), P(V_scored) for '{X_word}' and '{Y_word}'")

        prob_data = P_KV_scored_head[row_tuple[0]][row_tuple[1]]["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["xpos"]

        sns.lineplot(x=xpos, y=P_X, label='P(K)', ax=axs[i, 2])
        sns.lineplot(x=ypos, y=P_Y, label='P(V_scored)', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"P(K), P(V_scored) for '{X_word}' and '{Y_word}'")


    fig.suptitle(f"PDF of Q-K, Q-V_scored, K-V_scored for epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    row_tuple = word_indices[1]
    X_word = input_words[row_tuple[0]]
    K_prime = QKV_dict["K_head"]
    K_prime_row = K_prime[row_tuple[0]]
    prob_data = P_KV_scored_head[row_tuple[0]][row_tuple[1]]["prob_data"]
    P_K_prime = prob_data["P_X"]
    K_coords = prob_data["xpos"]
    dim_axis = np.arange(0, len(K_prime_row))

    N_bins = len(K_coords)
    ax.hist(K_prime_row, bins=N_bins, color='blue')
    ax.set_xlabel('Vector Dimension')
    ax.set_ylabel('Coordinate Value')
    ax.set_title(f"K' and P(K') for the word '{X_word}'")
    ax.legend()
    ax_twin = ax.twinx()
    ax_twin.plot(K_coords, P_K_prime, color='red', label='P_K\'')
    ax_twin.set_ylabel('Probability')
    plt.show(block=True)


def plot_softmax_distances(softmax_div_dict: dict, input_words: list, epoch: int, enc_layer: int, head_id: int, sentence_id: int) -> None:
    QK_head_JSD = softmax_div_dict["QK_head_JSD"]
    QV_scored_head_JSD = softmax_div_dict["QV_scored_head_JSD"]
    KV_scored_head_JSD = softmax_div_dict["KV_scored_head_JSD"]
    QK_head_bhattacharya = softmax_div_dict["QK_head_bhattacharya"]
    QV_scored_head_bhattacharya = softmax_div_dict["QV_scored_head_bhattacharya"]
    KV_scored_head_bhattacharya = softmax_div_dict["KV_scored_head_bhattacharya"]

    # ---------------------------------------------------------------------------------------------
    # Plot the Jensen-Shannon divergence between the rows of Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q-K head JSD distance matrix
    img = axs[0].imshow(QK_head_JSD, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(QK_head_JSD.shape[0]):
            for j in range(QK_head_JSD.shape[1]):
                text = axs[0].text(j, i, f"{QK_head_JSD[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"JSD between Q and K heads")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the Q-V_scored head JSD distance matrix
    img = axs[1].imshow(QV_scored_head_JSD, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(QV_scored_head_JSD.shape[0]):
            for j in range(QV_scored_head_JSD.shape[1]):
                text = axs[1].text(j, i, f"{QV_scored_head_JSD[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"JSD between the Q and V_scored heads")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K-V_scored head JSD distance matrix
    img = axs[2].imshow(KV_scored_head_JSD, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(KV_scored_head_JSD.shape[0]):
            for j in range(KV_scored_head_JSD.shape[1]):
                text = axs[2].text(j, i, f"{KV_scored_head_JSD[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"JSD between the K and V_scored heads")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"JSD between the rows of Q-K, Q-V_scored, K-V_scored heads\nfor epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

    # --------------------------------------------------------------------------------------------------------
    # Plot the Bhattacharya coefficient between the rows of Q-K head, Q-V_scored head, K-V_scored head
    # --------------------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q-K head Bhattacharya distance matrix
    img = axs[0].imshow(QK_head_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(QK_head_bhattacharya.shape[0]):
            for j in range(QK_head_bhattacharya.shape[1]):
                text = axs[0].text(j, i, f"{QK_head_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Bhattacharya coeff between Q-K heads")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the Q-V_scored head Bhattacharya distance matrix
    img = axs[1].imshow(QV_scored_head_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(QV_scored_head_bhattacharya.shape[0]):
            for j in range(QV_scored_head_bhattacharya.shape[1]):
                text = axs[1].text(j, i, f"{QV_scored_head_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Bhattacharya coeff between Q-V_scored heads")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K-V_scored head Bhattacharya distance matrix
    img = axs[2].imshow(KV_scored_head_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(KV_scored_head_bhattacharya.shape[0]):
            for j in range(KV_scored_head_bhattacharya.shape[1]):
                text = axs[2].text(j, i, f"{KV_scored_head_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Bhattacharya coeff between K-V_scored heads")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Bhattacharya coeff between the rows of Q-K, Q-V_scored and K-V_scored heads\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)


def plot_softmax_pmf(QKV_dict: dict, softmax_div_dict: dict, input_words: list, epoch: int, enc_layer: int, head_id: int, sentence_id: int) -> None:
    Q_head_probs = softmax_div_dict["Q_head_probs"]
    K_head_probs = softmax_div_dict["K_head_probs"]

    # Create the figure and axes
    # sns.set_theme(style="darkgrid", rc={'axes.facecolor':'lightcyan', 'figure.facecolor':'lightcyan'})
    sns.set_theme(style="darkgrid", rc={'axes.facecolor':'palegoldenrod', 'figure.facecolor':'white'})

    # Word indices to plot
    word_indices=[(4, 7), (5, 7), (11, 13), (4, 10), (10, 13), (7, 13), (0, 14), (0, 4), (7, 14), (4, 13), (10, 7), (1, 4)]

    fig, axs = plt.subplots(4,3)

    # ---------------------------------------------------------------------------------------------
    # Plot the probabilities of Q, K heads for specific words
    # ---------------------------------------------------------------------------------------------
    for i in range(12):
        a = i // 3
        b = i % 3
        row_tuple = word_indices[i]
        X_word = input_words[row_tuple[0]]
        Y_word = input_words[row_tuple[1]]
        P_X = Q_head_probs[row_tuple[0], :]
        P_Y = K_head_probs[row_tuple[1], :]
        dim_axis = np.arange(0, len(P_X))
        sns.lineplot(x=dim_axis, y=P_X, label=f'P(Q)', ax=axs[a, b])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P(K)', ax=axs[a, b])
        axs[a, b].legend()
        axs[a, b].set_xlabel('Vector Dimension')
        axs[a, b].set_ylabel('Probability')
        axs[a, b].set_title(f"P(Q): {X_word}, P(K): {Y_word}")

    fig.suptitle(f"Softmax probabilities of Q, K heads for epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K_head, P(K_head) for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    row_tuple = word_indices[1]
    X_word = input_words[row_tuple[0]]
    K_head_matrix = QKV_dict["K_head"]
    K_head_row = K_head_matrix[row_tuple[0]]
    P_K_head = K_head_probs[row_tuple[0]]
    dim_axis = np.arange(0, len(K_head_row))

    ax.bar(dim_axis, np.abs(K_head_row), color='blue')
    ax.set_xlabel('Vector Dimension')
    ax.set_ylabel('Coordinate Value')
    ax.set_title(f"K_head and P(K_head) for the word '{X_word}'")
    ax.legend()
    ax_twin = ax.twinx()
    ax_twin.plot(dim_axis, P_K_head, color='red')
    ax_twin.set_ylabel('Probability')
    plt.show(block=True)


def plot_KDE_probs(QKV_dict, dist_dict, input_words, epoch, enc_layer, head_id, sentence_id) -> None:
    QK_head_prob_list = dist_dict["QK_head_prob_matrix"]
    QV_scored_head_prob_list = dist_dict["QV_scored_head_prob_matrix"]
    KV_scored_head_prob_list = dist_dict["KV_scored_head_prob_matrix"]

    sns.set_theme(style="darkgrid", rc={'axes.facecolor':'palegoldenrod', 'figure.facecolor':'white'})

    # Word indices to plot
    word_indices=[4, 7, 10, 13]

    fig, axs = plt.subplots(4,3)

    # ---------------------------------------------------------------------------------------------
    # Plot the probabilities of Q-K, Q-V_scored, K-V_scored heads for specific words
    # ---------------------------------------------------------------------------------------------
    for i in range(4):
        row = word_indices[i]
        in_word = input_words[row]

        QK_head_prob_data = QK_head_prob_list[row][row]
        prob_data = QK_head_prob_data["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["ypos"]

        sns.lineplot(x=xpos, y=P_X, label=f'P(Q)', ax=axs[i, 0])
        sns.lineplot(x=ypos, y=P_Y, label=f'P(K)', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"Q, K probs for the word '{in_word}'")

        QV_scored_head_prob_data = QV_scored_head_prob_list[row][row]
        prob_data = QV_scored_head_prob_data["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["ypos"]

        sns.lineplot(x=xpos, y=P_X, label=f'P(Q)', ax=axs[i, 1])
        sns.lineplot(x=ypos, y=P_Y, label=f'P(V_scored)', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"Q, V_scored probs for word '{in_word}'")

        KV_scored_head_prob_data = KV_scored_head_prob_list[row][row]
        prob_data = KV_scored_head_prob_data["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["ypos"]

        sns.lineplot(x=xpos, y=P_X, label=f'P(K)', ax=axs[i, 2])
        sns.lineplot(x=ypos, y=P_Y, label=f'P(V_scored)', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"K, V_scored probs for word '{in_word}'")


    fig.suptitle(f"KDE probabilities of Q-K, Q-V_scored, K-V_scored for epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Q head, P(Q_head) and K head, P(K_head) and V_scored, P(V_scored) for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1, 3)
    row = word_indices[1]
    in_word = input_words[row]

    QV_scored_head_prob_data = QV_scored_head_prob_list[row][row]
    prob_data = QV_scored_head_prob_data["prob_data"]
    P_Q_head = prob_data["P_X"]
    xpos = prob_data["xpos"]
    P_V_scored_head = prob_data["P_Y"]
    ypos = prob_data["ypos"]
    KV_scored_head_prob_data = KV_scored_head_prob_list[row][row]
    prob_data = KV_scored_head_prob_data["prob_data"]
    P_K_head = prob_data["P_X"]
    xpos = prob_data["xpos"]

    Q_head_matrix = QKV_dict["Q_head"]
    K_head_matrix = QKV_dict["K_head"]
    V_scored_head_matrix = QKV_dict["V_scored_head"]

    Q_head = Q_head_matrix[row]
    K_head = K_head_matrix[row]
    V_scored_head = V_scored_head_matrix[row]

    # Plot Q head and P(Q_head)
    N_bins = P_Q_head.shape[0]
    axs[0].hist(Q_head, bins=N_bins, color='blue')
    axs[0].set_xlabel('Coordinate Values')
    axs[0].set_ylabel('Counts')
    axs[0].set_title("Q_head and P(Q_head)")
    ax_twin = axs[0].twinx()
    ax_twin.plot(xpos, P_Q_head, color='red')
    ax_twin.set_ylabel('Probability')

    # Plot K head and P(K_head)
    N_bins = P_K_head.shape[0]
    axs[1].hist(K_head, bins=N_bins, color='blue')
    axs[1].set_xlabel('Coordinate Values')
    axs[1].set_ylabel('Counts')
    axs[1].set_title("K_head and P(K_head)")
    ax_twin = axs[1].twinx()
    ax_twin.plot(xpos, P_K_head, color='red')
    ax_twin.set_ylabel('Probability')

    # Plot V_scored head and P(V_scored)
    N_bins = P_V_scored_head.shape[0]
    axs[2].hist(V_scored_head, bins=N_bins, color='blue')
    axs[2].set_xlabel('Coordinate Values')
    axs[2].set_ylabel('Counts')
    axs[2].set_title("V_scored_head and P(V_scored)")
    ax_twin = axs[2].twinx()
    ax_twin.plot(xpos, P_V_scored_head, color='red')
    ax_twin.set_ylabel('Probability')

    fig.suptitle(f"KDE probabilities of Q, K and V_scored for epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
    plt.show(block=True)

def plot_MI_WSD_scatter(MI_WSD_dict: dict, input_words: list, epoch: int, enc_layer: int, head_id: int, sentence_id: int) -> None:
    MI_QK_head = MI_WSD_dict["MI_QK_head"]
    MI_QV_scored_head = MI_WSD_dict["MI_QV_scored_head"]
    MI_KV_scored_head = MI_WSD_dict["MI_KV_scored_head"]
    WSD_QK_head = MI_WSD_dict["WSD_QK_head"]
    WSD_QV_scored_head = MI_WSD_dict["WSD_QV_scored_head"]
    WSD_KV_scored_head = MI_WSD_dict["WSD_KV_scored_head"]

    # Word indices to plot
    word_indices=[4, 7, 10, 13]
    word_colors = [colors.cnames['gray'],      # SOS
                   colors.cnames['brown'],     # There
                   colors.cnames['yellow'],    # is
                   colors.cnames['orange'],    # a
                   colors.cnames['red'],       # dog
                   colors.cnames['black'],     # in
                   colors.cnames['purple'],    # the
                   colors.cnames['green'],     # car
                   colors.cnames['cyan'],      # and
                   colors.cnames['pink'],      # a
                   colors.cnames['magenta'],   # cat
                   colors.cnames['olive'],     # in
                   colors.cnames['lime'],      # the
                   colors.cnames['blue'],      # van
                   colors.cnames['peru']]      # EOS

    for word_idx in word_indices:
        sns.set_theme(style="darkgrid")
        fig, axs = plt.subplots(1,3)
        axes = axs.ravel()
        word_str = input_words[word_idx]
        plot_words = input_words.copy()
        plot_words.remove(word_str)

        for ax_idx, ax in enumerate(axes):
            match ax_idx:
                case 0:
                    # Plot Q-K head MI vs Q-K head Wasserstein distance
                    annotate_text = []
                    x = MI_QK_head[word_idx, :]
                    y = WSD_QK_head[word_idx, :]
                    x = np.delete(x, word_idx)
                    y = np.delete(y, word_idx)
                    ax.scatter(x, y, c=word_colors[:-1])
                    ax.set_xlabel('Mutual Information (bits)')
                    ax.set_ylabel('Wasserstein Distance')
                    ax.set_title(f"Q-K head probability space for the word '{word_str}'")
                    for i, txt in enumerate(plot_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

                case 1:
                    # Plot Q-V_scored head MI vs Q-V_scored head Wasserstein distance
                    annotate_text = []
                    x = MI_QV_scored_head[word_idx, :]
                    y = WSD_QV_scored_head[word_idx, :]
                    x = np.delete(x, word_idx)
                    y = np.delete(y, word_idx)
                    ax.scatter(x, y, c=word_colors[:-1])
                    ax.set_xlabel('Mutual Information (bits)')
                    ax.set_ylabel('Wasserstein Distance')
                    ax.set_title(f"Q-V_scored probability space for the word '{word_str}'")
                    for i, txt in enumerate(plot_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

                case 2:
                    # Plot K-V_scored head MI vs K-V_scored head Wasserstein distance
                    annotate_text = []
                    x = MI_KV_scored_head[word_idx, :]
                    y = WSD_KV_scored_head[word_idx, :]
                    x = np.delete(x, word_idx)
                    y = np.delete(y, word_idx)
                    ax.scatter(x, y, c=word_colors[:-1])
                    ax.set_xlabel('Mutual Information (bits)')
                    ax.set_ylabel('Wasserstein Distance')
                    ax.set_title(f"K-V_scored probability space for the word '{word_str}'")
                    for i, txt in enumerate(plot_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

        fig.suptitle(f"Probability space for Q-K, Q-V_scored and K-V_scored heads\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
        plt.show(block=True)


def plot_BC_JSD_scatter(softmax_div_dict: dict, input_words: list, epoch: int, enc_layer: int, head_id: int, sentence_id: int) -> None:
    QK_head_JSD = softmax_div_dict["QK_head_JSD"]
    QV_scored_head_JSD = softmax_div_dict["QV_scored_head_JSD"]
    KV_scored_head_JSD = softmax_div_dict["KV_scored_head_JSD"]
    QK_head_bhattacharya = softmax_div_dict["QK_head_bhattacharya"]
    QV_scored_head_bhattacharya = softmax_div_dict["QV_scored_head_bhattacharya"]
    KV_scored_head_bhattacharya = softmax_div_dict["KV_scored_head_bhattacharya"]

    # Word indices to plot
    word_indices=[4, 7, 10, 13]
    word_colors = [colors.cnames['gray'],      # SOS
                   colors.cnames['brown'],     # There
                   colors.cnames['yellow'],    # is
                   colors.cnames['orange'],    # a
                   colors.cnames['red'],       # dog
                   colors.cnames['black'],     # in
                   colors.cnames['purple'],    # the
                   colors.cnames['green'],     # car
                   colors.cnames['cyan'],      # and
                   colors.cnames['pink'],      # a
                   colors.cnames['magenta'],   # cat
                   colors.cnames['olive'],     # in
                   colors.cnames['lime'],      # the
                   colors.cnames['blue'],      # van
                   colors.cnames['peru']]      # EOS


    for word_idx in word_indices:
        sns.set_theme(style="darkgrid")
        fig, axs = plt.subplots(1,3)
        axes = axs.ravel()
        word_str = input_words[word_idx]

        for ax_idx, ax in enumerate(axes):
            match ax_idx:
                case 0:
                    # Plot Q' BC vs Q' JSD
                    annotate_text = []
                    x = QK_head_bhattacharya[word_idx, :]
                    y = QK_head_JSD[word_idx, :]
                    ax.scatter(x, y, c=word_colors)
                    ax.set_xlabel('Bhattachrya Coefficient')
                    ax.set_ylabel('Jensen-Shannon Divergence')
                    ax.set_title(f"Q-K probability space for the word '{word_str}'")
                    for i, txt in enumerate(input_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

                case 1:
                    # Plot K' BC vs K' JSD
                    annotate_text = []
                    x = QV_scored_head_bhattacharya[word_idx, :]
                    y = QV_scored_head_JSD[word_idx, :]
                    ax.scatter(x, y, c=word_colors)
                    ax.set_xlabel('Bhattachrya Coefficient')
                    ax.set_ylabel('Jesnen-Shannon Divergence')
                    ax.set_title(f"K' probability space for the word '{word_str}'")
                    for i, txt in enumerate(input_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

                case 2:
                    # Plot V' BC vs V' JSD
                    annotate_text = []
                    x = KV_scored_head_bhattacharya[word_idx, :]
                    y = KV_scored_head_JSD[word_idx, :]
                    ax.scatter(x, y, c=word_colors)
                    ax.set_xlabel('Bhattachrya Coefficient')
                    ax.set_ylabel('Jesnen-Shannon Divergence')
                    ax.set_title(f"V' probability space for the word '{word_str}'")
                    for i, txt in enumerate(input_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

        fig.suptitle(f"Probability space for Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, head {head_id}, sentence {sentence_id}")
        plt.show(block=True)

# ---------------------------------- Plotting routines end ----------------------------------


def process_QKV_head_softmax_distances(QKV_dict: dict) -> dict:
    # Extract the Q, K head probes
    Q_head = QKV_dict["Q_head"]
    K_head = QKV_dict["K_head"]
    V_scored_head = QKV_dict["V_scored_head"]

    # Parameters of the processing function
    N_rows = Q_head.shape[0]
    beta = 2.5
    # PROB_THRESHOLD = 0.0015
    PROB_THRESHOLD = 0

    # Initialize the Q' probabilty matrix
    Q_head_probs = np.zeros(Q_head.shape)
    K_head_probs = np.zeros(K_head.shape)
    V_scored_head_probs = np.zeros(V_scored_head.shape)

    for i in range(N_rows):
        # Compute P_Q softmax
        X = np.abs(Q_head[i])
        # X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        Q_head_probs[i, :] = P_X
        # Compute P_K softmax
        Y = np.abs(K_head[i])
        # Y /= max(Y)
        P_Y = np.exp(beta * Y)/np.sum(np.exp(beta * Y))
        K_head_probs[i, :] = P_Y
        # Compute P_V_scored softmax
        Z = np.abs(V_scored_head[i])
        # Z /= max(Z)
        P_Z = np.exp(beta * Z)/np.sum(np.exp(beta * Z))
        V_scored_head_probs[i, :] = P_Z

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_Q_head_probs = Q_head_probs.copy()
    idx = np.where(Q_head_probs < PROB_THRESHOLD)
    tmp_Q_head_probs[idx] = 0

    tmp_K_head_probs = K_head_probs.copy()
    idx = np.where(K_head_probs < PROB_THRESHOLD)
    tmp_K_head_probs[idx] = 0

    tmp_V_scored_head_probs = V_scored_head_probs.copy()
    idx = np.where(V_scored_head_probs < PROB_THRESHOLD)
    tmp_V_scored_head_probs[idx] = 0

    # Compute the Jensen-Shannon diveregence between the Q-K, Q-V_scored, K-V_scored heads
    QK_head_JSD = compute_jensen_shannon_divergence(tmp_Q_head_probs, tmp_K_head_probs)
    QV_scored_head_JSD = compute_jensen_shannon_divergence(tmp_Q_head_probs, tmp_V_scored_head_probs)
    KV_scored_head_JSD = compute_jensen_shannon_divergence(tmp_K_head_probs, tmp_V_scored_head_probs)

    # Compute the Bhattacharya coefficients between the Q-K, Q-V_scored, K-V_scored heads
    QK_head_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_head_probs, tmp_K_head_probs)
    QV_scored_head_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_head_probs, tmp_V_scored_head_probs)
    KV_scored_head_bhattacharya = compute_bhattacharya_coefficient(tmp_K_head_probs, tmp_V_scored_head_probs)

    return dict(Q_head_probs=Q_head_probs, 
                K_head_probs=K_head_probs,
                V_scored_head_probs=V_scored_head_probs,
                QK_head_JSD=QK_head_JSD,
                QV_scored_head_JSD=QV_scored_head_JSD,
                KV_scored_head_JSD=KV_scored_head_JSD,
                QK_head_bhattacharya=QK_head_bhattacharya,
                QV_scored_head_bhattacharya=QV_scored_head_bhattacharya,
                KV_scored_head_bhattacharya=KV_scored_head_bhattacharya)


def process_QKV_head_KDE_distances(KDE_dict: dict) -> dict:
    # PROB_THRESHOLD = 0.0015
    PROB_THRESHOLD = 0

    # =================================================================================================
    # Process Q' KDE data
    # =================================================================================================
    Q_prime_prob_matrix = KDE_dict["Q_prime_prob_matrix"]
    N_rows = len(Q_prime_prob_matrix)
    data = Q_prime_prob_matrix[0][0]
    prob_data = data["prob_data"]
    P_Q = prob_data["P_X"]

    Q_prime_probs = np.zeros((N_rows, P_Q.shape[0]))
    Q_prime_coord = list()
    for i in range(N_rows):
        data = Q_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_Q = prob_data["P_X"]
        Q_prime_probs[i, :] = P_Q
        Q_prime_coord.append(prob_data["xpos"])

    tmp_Q_prime_probs = Q_prime_probs.copy()
    idx = np.where(Q_prime_probs < PROB_THRESHOLD)
    tmp_Q_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the Q' matrices
    Q_prime_wdist = compute_wasserstein_distance(tmp_Q_prime_probs, tmp_Q_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the Q' matrices
    Q_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_prime_probs, tmp_Q_prime_probs)

    # =================================================================================================
    # Process K' KDE data
    # =================================================================================================
    K_prime_prob_matrix = KDE_dict["K_prime_prob_matrix"]
    N_rows = len(K_prime_prob_matrix)
    data = K_prime_prob_matrix[0][0]
    prob_data = data["prob_data"]
    P_K = prob_data["P_X"]

    K_prime_probs = np.zeros((N_rows, P_K.shape[0]))
    K_prime_coord = list()
    for i in range(N_rows):
        data = K_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_K = prob_data["P_X"]
        K_prime_probs[i, :] = P_K
        K_prime_coord.append(prob_data["xpos"])

    tmp_K_prime_probs = K_prime_probs.copy()
    idx = np.where(K_prime_probs < PROB_THRESHOLD)
    tmp_K_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the K' matrices
    K_prime_wdist = compute_wasserstein_distance(tmp_K_prime_probs, tmp_K_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the K' matrices
    K_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_K_prime_probs, tmp_K_prime_probs)

    # =================================================================================================
    # Process V' KDE data
    # =================================================================================================
    V_prime_prob_matrix = KDE_dict["V_prime_prob_matrix"]
    N_rows = len(V_prime_prob_matrix)
    data = V_prime_prob_matrix[0][0]
    prob_data = data["prob_data"]
    P_V = prob_data["P_X"]

    V_prime_probs = np.zeros((N_rows, P_V.shape[0]))
    V_prime_coord = list()
    for i in range(N_rows):
        data = V_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_V = prob_data["P_X"]
        V_prime_probs[i, :] = P_V
        V_prime_coord.append(prob_data["xpos"])

    tmp_V_prime_probs = V_prime_probs.copy()
    idx = np.where(V_prime_probs < PROB_THRESHOLD)
    tmp_V_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the V' matrices
    V_prime_wdist = compute_wasserstein_distance(tmp_V_prime_probs, tmp_V_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the V' matrices
    V_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_V_prime_probs, tmp_V_prime_probs)

    return dict(Q_prime_probs=Q_prime_probs, 
                Q_prime_coord=Q_prime_coord,
                Q_prime_wdist=Q_prime_wdist, 
                Q_prime_bhattacharya=Q_prime_bhattacharya,
                K_prime_probs=K_prime_probs,
                K_prime_coord=K_prime_coord,
                K_prime_wdist=K_prime_wdist,
                K_prime_bhattacharya=K_prime_bhattacharya,
                V_prime_probs=V_prime_probs,
                V_prime_coord=V_prime_coord,
                V_prime_wdist=V_prime_wdist,
                V_prime_bhattacharya=V_prime_bhattacharya)


def process_QKV_head_MI_WSD(QKV_dict: dict) -> dict:
    # Extract the Q', K', V' probes
    Q_head = QKV_dict["Q_head"]
    K_head = QKV_dict["K_head"]
    V_scored_head = QKV_dict["V_scored_head"]

    print("\nComputing the mutual information between the Q and K heads ...")
    P_QK_head, MI_QK_head, WSD_QK_head = KDE_mutual_info_WSD(Q_head, K_head, False)

    print("\nComputing the mutual information between the Q and V scored heads ...")
    P_QV_scored_head, MI_QV_scored_head, WSD_QV_scored_head = KDE_mutual_info_WSD(Q_head, V_scored_head, False)

    print("\nComputing the mutual information between the K and V scored heads ...")
    P_KV_scored_head, MI_KV_scored_head, WSD_KV_scored_head = KDE_mutual_info_WSD(K_head, V_scored_head, False)

    return dict(P_QK_head=P_QK_head,
                MI_QK_head=MI_QK_head, 
                WSD_QK_head=WSD_QK_head,
                P_QV_scored_head=P_QV_scored_head,
                MI_QV_scored_head=MI_QV_scored_head, 
                WSD_QV_scored_head=WSD_QV_scored_head,
                P_KV_scored_head=P_KV_scored_head,
                MI_KV_scored_head=MI_KV_scored_head,
                WSD_KV_scored_head=WSD_KV_scored_head)

def process_QKV_head_KDE_MI(QKV_dict: dict) -> dict:
    # Extract the Q, K, V_scored head probes
    Q_head = QKV_dict["Q_head"]
    K_head = QKV_dict["K_head"]
    V_scored_head = QKV_dict["V_scored_head"]

    print("\nComputing the mutual information between the rows of Q and K heads ...")
    QK_head_prob_matrix, MI_QK_head = KDE_mutual_info(Q_head, K_head, True)

    print("\nComputing the mutual information between the rows of Q and V scored heads ...")
    QV_scored_head_prob_matrix, MI_QV_scored_head = KDE_mutual_info(Q_head, V_scored_head, True)

    print("\nComputing the mutual information between the rows of K and V scored heads ...")
    KV_scored_head_prob_matrix, MI_KV_scored_head = KDE_mutual_info(K_head, V_scored_head, True)

    return dict(QK_head_prob_matrix=QK_head_prob_matrix, 
                MI_QK_head=MI_QK_head,
                QV_scored_head_prob_matrix=QV_scored_head_prob_matrix,
                MI_QV_scored_head=MI_QV_scored_head,
                KV_scored_head_prob_matrix=KV_scored_head_prob_matrix,
                MI_KV_head=MI_KV_scored_head)

def analyze_encoder_QKV_head(analyzer) -> None:
    # Parameters for the analysis
    epoch = 19
    enc_layer = 0
    sentence_id = 3
    head_id = 0

    # For the specific epoch load all the encoder probes from the disk
    analyzer.load_encoder_probes(epoch)

    # Number of input sentences in this epoch
    N_input_sentences = len(analyzer.encoder_probe._probe_in)
    print(f"Number of input sentences: {N_input_sentences}")

    # Number of input tokens for the specific sentence
    N_input_tokens = analyzer.get_encoder_input_token_count(sentence_id)
    print(f"Number of input tokens: {N_input_tokens}")

    # Get the words for the specific input sentence
    input_words = analyzer.get_encoder_input_words(sentence_id)

    # Get the Q, K, V attention head probes for the specific layer
    QKV_dict = analyzer.get_encoder_QKV_head(enc_layer, head_id, sentence_id, N_input_tokens)

    # Get the attention scores for the specific encoder layer and head
    atten_scores =  analyzer.get_encoder_attention_scores(enc_layer, head_id, sentence_id, N_input_tokens)

    # Process the Q, K, V_scores attention head probes to compute the Mutual Information
    data_filename = Path(f"data/MI_WSD_head_dict_epoch_{epoch}_layer_{enc_layer}_head_{head_id}_sentence_{sentence_id}.pt")
    if data_filename.exists():
        print(f"Loading the MI_WSD data from {data_filename}")
        MI_WSD_dict = torch.load(data_filename, weights_only=False)
    else:
        MI_WSD_dict = process_QKV_head_MI_WSD(QKV_dict)
        torch.save(MI_WSD_dict, data_filename)

    # Plot the Q-K, V_scored mutual information matrices and the attention scores
    plot_QKV_head_MI_WSD_atten_scores(MI_WSD_dict, atten_scores, input_words, epoch, enc_layer, head_id, sentence_id)

    # Plot the PDF of Q-K, Q-V_scored, K-V_scored for specific words
    plot_QKV_head_pdf(QKV_dict, MI_WSD_dict, input_words, epoch, enc_layer, head_id, sentence_id)

    # Process the Q, K and V_scoredhead matrices to obtain the softmax 
    # probabilities of # each row vector and the Wasserstein distance 
    # and Bhattacharya coeffs # between the rows of Q and K heads and 
    # also the Q-V_scored and K-V_scored heads
    softmax_div_dict = process_QKV_head_softmax_distances(QKV_dict)

    # Plot the JSD and Bhattacharya coefficients between the rows of Q-K, Q-V_scored, K-V_scored heads
    plot_softmax_distances(softmax_div_dict, input_words, epoch, enc_layer, head_id, sentence_id)

    # Plot the PMF based on softmax probabilities for Q, K, V_scored heads
    plot_softmax_pmf(QKV_dict, softmax_div_dict, input_words, epoch, enc_layer, head_id, sentence_id)

    # Plot the Mutual Information and the Wasserstein distances as a scatter plot
    plot_MI_WSD_scatter(MI_WSD_dict, input_words, epoch, enc_layer, head_id, sentence_id)

    # Plot the Bhattachrya coefficient and Jensen-Shannon divergence as a scatter plot
    plot_BC_JSD_scatter(softmax_div_dict, input_words, epoch, enc_layer, head_id, sentence_id)

