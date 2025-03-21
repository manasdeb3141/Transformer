
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


# Class/functions implemented by this application
from probability_distances import compute_bhattacharya_coefficient
from probability_distances import compute_jensen_shannon_divergence
from compute_MI import KDE_mutual_info_WSD


# ---------------------------------- Plotting routines start --------------------------------

def plot_QKV_prime_MI_WSD(MI_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int):
    # Extract the mutual information matrices
    MI_Q_prime_orig = MI_dict["MI_Q_prime"]
    MI_K_prime_orig = MI_dict["MI_K_prime"]
    MI_V_prime_orig = MI_dict["MI_V_prime"]

    # Make copied of these matrices as we will zero out the diagonals
    # for plotting but will need the original matrices for other calculations
    MI_Q_prime = np.copy(MI_Q_prime_orig)
    MI_K_prime = np.copy(MI_K_prime_orig)
    MI_V_prime = np.copy(MI_V_prime_orig)

    # Fill the diagonal with zeros so that the plot is not dominated by the diagonal values
    # The diagonals are just entropies of the individual rows
    np.fill_diagonal(MI_Q_prime, 0)
    np.fill_diagonal(MI_K_prime, 0)
    np.fill_diagonal(MI_V_prime, 0)

    # Flag to display the mutual information values on the plot
    display_values = False

    # Create the figure and axes
    fig, axs = plt.subplots(1,3)

    # Plot the Q' mutual information matrix
    img = axs[0].imshow(MI_Q_prime, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_Q_prime.shape[0]):
            for j in range(MI_Q_prime.shape[1]):
                text = axs[0].text(j, i, f"{MI_Q_prime[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K' mutual information matrix
    img = axs[1].imshow(MI_K_prime, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_K_prime.shape[0]):
            for j in range(MI_K_prime.shape[1]):
                text = axs[1].text(j, i, f"{MI_K_prime[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"MI between rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the V' mutual information matrix
    img = axs[2].imshow(MI_V_prime, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_V_prime.shape[0]):
            for j in range(MI_V_prime.shape[1]):
                text = axs[2].text(j, i, f"{MI_V_prime[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"MI between rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Mutual Information between the rows of Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Wasserstein distance between the rows of Q', K', V'
    # ---------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1,3)

    Q_prime_wdist = MI_dict["WSD_Q_prime"]
    K_prime_wdist = MI_dict["WSD_K_prime"]
    V_prime_wdist = MI_dict["WSD_V_prime"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q' Wasserstein distance matrix
    img = axs[0].imshow(Q_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(Q_prime_wdist.shape[0]):
            for j in range(Q_prime_wdist.shape[1]):
                text = axs[0].text(j, i, f"{Q_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Wasserstein distance between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K' Wasserstein distance matrix
    img = axs[1].imshow(K_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(K_prime_wdist.shape[0]):
            for j in range(K_prime_wdist.shape[1]):
                text = axs[1].text(j, i, f"{K_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Wasserstein distance between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the V' Wasserstein distance matrix
    img = axs[2].imshow(V_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(V_prime_wdist.shape[0]):
            for j in range(V_prime_wdist.shape[1]):
                text = axs[2].text(j, i, f"{V_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Wasserstein distance between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Wasserstein distance between the rows of Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)


def plot_QKV_prime_pdf(QKV_dict: dict, MI_WSD_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_prime_probs = MI_WSD_dict["P_Q_prime"]
    K_prime_probs = MI_WSD_dict["P_K_prime"]
    V_prime_probs = MI_WSD_dict["P_V_prime"]

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

        prob_data = Q_prime_probs[row_tuple[0]][row_tuple[1]]["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["xpos"]

        sns.lineplot(x=xpos, y=P_X, label=f'P({X_word})', ax=axs[i, 0])
        sns.lineplot(x=ypos, y=P_Y, label=f'P({Y_word})', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"Q' probabilities")

        prob_data = K_prime_probs[row_tuple[0]][row_tuple[1]]["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["xpos"]

        sns.lineplot(x=xpos, y=P_X, label=f'P({X_word})', ax=axs[i, 1])
        sns.lineplot(x=ypos, y=P_Y, label=f'P({Y_word})', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"K' probabilities")

        prob_data = V_prime_probs[row_tuple[0]][row_tuple[1]]["prob_data"]
        P_X = prob_data["P_X"]
        xpos = prob_data["xpos"]
        P_Y = prob_data["P_Y"]
        ypos = prob_data["xpos"]

        sns.lineplot(x=xpos, y=P_X, label=f'P({X_word})', ax=axs[i, 2])
        sns.lineplot(x=ypos, y=P_Y, label=f'P({Y_word})', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"V' probabilities")


    fig.suptitle(f"PDF of Q', K', V' for epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    row_tuple = word_indices[1]
    X_word = input_words[row_tuple[0]]
    K_prime = QKV_dict["K_prime"]
    K_prime_row = K_prime[row_tuple[0]]
    prob_data = K_prime_probs[row_tuple[0]][row_tuple[1]]["prob_data"]
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


def plot_softmax_pmf(QKV_dict: dict, softmax_div_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_prime_probs = softmax_div_dict["Q_prime_probs"]
    K_prime_probs = softmax_div_dict["K_prime_probs"]
    V_prime_probs = softmax_div_dict["V_prime_probs"]

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

        P_X = Q_prime_probs[row_tuple[0]]
        P_Y = Q_prime_probs[row_tuple[1]]
        dim_axis = np.arange(0, len(P_X))

        sns.lineplot(x=dim_axis, y=P_X, label=f'P({X_word})', ax=axs[i, 0])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P({Y_word})', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"Q' probabilities")

        P_X = K_prime_probs[row_tuple[0]]
        P_Y = K_prime_probs[row_tuple[1]]
        dim_axis = np.arange(0, len(P_X))

        sns.lineplot(x=dim_axis, y=P_X, label=f'P({X_word})', ax=axs[i, 1])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P({Y_word})', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"K' probabilities")

        P_X = V_prime_probs[row_tuple[0]]
        P_Y = V_prime_probs[row_tuple[1]]
        dim_axis = np.arange(0, len(P_X))

        sns.lineplot(x=dim_axis, y=P_X, label=f'P({X_word})', ax=axs[i, 2])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P({Y_word})', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"V' probabilities")


    fig.suptitle(f"PMF of Q', K', V' for epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    row_tuple = word_indices[1]
    X_word = input_words[row_tuple[0]]
    K_prime = QKV_dict["K_prime"]
    K_prime_row = K_prime[row_tuple[0]]
    P_K_prime = K_prime_probs[row_tuple[0]]
    dim_axis = np.arange(0, len(K_prime_row))

    ax.bar(dim_axis, np.abs(K_prime_row), color='blue')
    ax.set_xlabel('Vector Dimension')
    ax.set_ylabel('Coordinate Value')
    ax.set_title(f"K' and P(K') for the word '{X_word}'")
    ax.legend()
    ax_twin = ax.twinx()
    ax_twin.plot(dim_axis, P_K_prime, color='red', label='P_K\'')
    ax_twin.set_ylabel('Probability')
    plt.show(block=True)


def plot_softmax_distances(softmax_div_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_prime_JSD = softmax_div_dict["Q_prime_JSD"]
    K_prime_JSD = softmax_div_dict["K_prime_JSD"]
    V_prime_JSD = softmax_div_dict["V_prime_JSD"]
    Q_prime_bhattacharya = softmax_div_dict["Q_prime_bhattacharya"]
    K_prime_bhattacharya = softmax_div_dict["K_prime_bhattacharya"]
    V_prime_bhattacharya = softmax_div_dict["V_prime_bhattacharya"]

    # ---------------------------------------------------------------------------------------------
    # Plot the Jensen-Shannon divergence between the rows of Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q' JSD distance matrix
    img = axs[0].imshow(Q_prime_JSD, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(Q_prime_JSD.shape[0]):
            for j in range(Q_prime_JSD.shape[1]):
                text = axs[0].text(j, i, f"{Q_prime_JSD[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"JSD between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K' JSD distance matrix
    img = axs[1].imshow(K_prime_JSD, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(K_prime_JSD.shape[0]):
            for j in range(K_prime_JSD.shape[1]):
                text = axs[1].text(j, i, f"{K_prime_JSD[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"JSD between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the V' JSD distance matrix
    img = axs[2].imshow(V_prime_JSD, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(V_prime_JSD.shape[0]):
            for j in range(V_prime_JSD.shape[1]):
                text = axs[2].text(j, i, f"{V_prime_JSD[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"JSD between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"JSD between the rows of Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Bhattacharya coefficient between the rows of Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q' Bhattacharya distance matrix
    img = axs[0].imshow(Q_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(Q_prime_bhattacharya.shape[0]):
            for j in range(Q_prime_bhattacharya.shape[1]):
                text = axs[0].text(j, i, f"{Q_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Bhattacharya coeff between the rows of Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the K' Bhattacharya distance matrix
    img = axs[1].imshow(K_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(K_prime_bhattacharya.shape[0]):
            for j in range(K_prime_bhattacharya.shape[1]):
                text = axs[1].text(j, i, f"{K_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Bhattacharya coeff between the rows of K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the V' Bhattacharya distance matrix
    img = axs[2].imshow(V_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(V_prime_bhattacharya.shape[0]):
            for j in range(V_prime_bhattacharya.shape[1]):
                text = axs[2].text(j, i, f"{V_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Bhattacharya coeff between the rows of V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Bhattacharya coeff between the rows of Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

def plot_MI_WSD_scatter(MI_WSD_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    MI_Q_prime = MI_WSD_dict["MI_Q_prime"]
    MI_K_prime = MI_WSD_dict["MI_K_prime"]
    MI_V_prime = MI_WSD_dict["MI_V_prime"]
    WSD_Q_prime = MI_WSD_dict["WSD_Q_prime"]
    WSD_K_prime = MI_WSD_dict["WSD_K_prime"]
    WSD_V_prime = MI_WSD_dict["WSD_V_prime"]

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
                    # Plot Q' MI vs Q' Wasserstein distance
                    annotate_text = []
                    x = MI_Q_prime[word_idx, :]
                    y = WSD_Q_prime[word_idx, :]
                    x = np.delete(x, word_idx)
                    y = np.delete(y, word_idx)
                    ax.scatter(x, y, c=word_colors[:-1])
                    ax.set_xlabel('Mutual Information (bits)')
                    ax.set_ylabel('Wasserstein Distance')
                    ax.set_title(f"Q' probability space for the word '{word_str}'")
                    for i, txt in enumerate(plot_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

                case 1:
                    # Plot K' MI vs K' Wasserstein distance
                    annotate_text = []
                    x = MI_K_prime[word_idx, :]
                    y = WSD_K_prime[word_idx, :]
                    x = np.delete(x, word_idx)
                    y = np.delete(y, word_idx)
                    ax.scatter(x, y, c=word_colors[:-1])
                    ax.set_xlabel('Mutual Information (bits)')
                    ax.set_ylabel('Wasserstein Distance')
                    ax.set_title(f"K' probability space for the word '{word_str}'")
                    for i, txt in enumerate(plot_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

                case 2:
                    # Plot V' Bhattacharya vs V' Wasserstein distance
                    annotate_text = []
                    x = MI_V_prime[word_idx, :]
                    y = WSD_V_prime[word_idx, :]
                    x = np.delete(x, word_idx)
                    y = np.delete(y, word_idx)
                    ax.scatter(x, y, c=word_colors[:-1])
                    ax.set_xlabel('Mutual Information (bits)')
                    ax.set_ylabel('Wasserstein Distance')
                    ax.set_title(f"V' probability space for the word '{word_str}'")
                    for i, txt in enumerate(plot_words):
                        annotate_text.append(ax.text(x[i], y[i], txt, ha='center', va='center'))
                    adjust_text(annotate_text, 
                                autoalign='y',
                                only_move={'points':'', 'text':'xy'},
                                arrowprops=dict(arrowstyle="->", color='k', lw=0.5),
                                ax=ax)

        fig.suptitle(f"Probability space for Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
        plt.show(block=True)


def plot_BC_JSD_scatter(softmax_div_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_prime_JSD = softmax_div_dict["Q_prime_JSD"]
    K_prime_JSD = softmax_div_dict["K_prime_JSD"]
    V_prime_JSD = softmax_div_dict["V_prime_JSD"]
    Q_prime_bhattacharya = softmax_div_dict["Q_prime_bhattacharya"]
    K_prime_bhattacharya = softmax_div_dict["K_prime_bhattacharya"]
    V_prime_bhattacharya = softmax_div_dict["V_prime_bhattacharya"]

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
                    x = Q_prime_bhattacharya[word_idx, :]
                    y = Q_prime_JSD[word_idx, :]
                    ax.scatter(x, y, c=word_colors)
                    ax.set_xlabel('Bhattachrya Coefficient')
                    ax.set_ylabel('Jensen-Shannon Divergence')
                    ax.set_title(f"Q' probability space for the word '{word_str}'")
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
                    x = K_prime_bhattacharya[word_idx, :]
                    y = K_prime_JSD[word_idx, :]
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
                    x = V_prime_bhattacharya[word_idx, :]
                    y = V_prime_JSD[word_idx, :]
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

        fig.suptitle(f"Probability space for Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
        plt.show(block=True)


# ---------------------------------- Plotting routines end ----------------------------------


def process_QKV_prime_softmax_distances(QKV_dict: dict) -> dict:
    # Extract the Q', K', V' probes
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    # Parameters of the processing function
    N_rows = Q_prime.shape[0]
    beta = 2.5
    # PROB_THRESHOLD = 0.0015
    PROB_THRESHOLD = 0

    # Initialize the Q' probabilty matrix
    Q_prime_probs = np.zeros(Q_prime.shape)

    for i in range(N_rows):
        X = np.abs(Q_prime[i])
        # X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        Q_prime_probs[i, :] = P_X

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_Q_prime_probs = Q_prime_probs.copy()
    idx = np.where(Q_prime_probs < PROB_THRESHOLD)
    tmp_Q_prime_probs[idx] = 0

    # Compute the Jensen-Shannon divergence between the rows of the Q' matrices
    Q_prime_JSD = compute_jensen_shannon_divergence(tmp_Q_prime_probs, tmp_Q_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the Q' matrices
    Q_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_prime_probs, tmp_Q_prime_probs)

    # Initialize the K' probabilty matrix
    K_prime_probs = np.zeros(K_prime.shape)

    for i in range(N_rows):
        X = np.abs(K_prime[i])
        # X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        K_prime_probs[i, :] = P_X

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_K_prime_probs = K_prime_probs.copy()
    idx = np.where(K_prime_probs < PROB_THRESHOLD)
    tmp_K_prime_probs[idx] = 0

    # Compute the Jensen-Shannon divergence between the rows of the K' matrices
    K_prime_JSD = compute_jensen_shannon_divergence(tmp_K_prime_probs, tmp_K_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the K' matrices
    K_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_K_prime_probs, tmp_K_prime_probs)

    # Initialize the V' probabilty matrix
    V_prime_probs = np.zeros(V_prime.shape)

    for i in range(N_rows):
        X = np.abs(V_prime[i])
        # X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        V_prime_probs[i, :] = P_X

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_V_prime_probs = V_prime_probs.copy()
    idx = np.where(V_prime_probs < PROB_THRESHOLD)
    tmp_V_prime_probs[idx] = 0

    # Compute the Jensen-Shannon divergence between the rows of the V' matrices
    V_prime_JSD = compute_jensen_shannon_divergence(tmp_V_prime_probs, tmp_V_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the V' matrices
    V_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_V_prime_probs, tmp_V_prime_probs)

    return dict(Q_prime_probs=Q_prime_probs, 
                K_prime_probs=K_prime_probs,
                V_prime_probs=V_prime_probs, 
                Q_prime_JSD=Q_prime_JSD, 
                K_prime_JSD=K_prime_JSD, 
                V_prime_JSD=V_prime_JSD,
                Q_prime_bhattacharya=Q_prime_bhattacharya,
                K_prime_bhattacharya=K_prime_bhattacharya,
                V_prime_bhattacharya=V_prime_bhattacharya)

def process_QKV_prime_MI_WSD(QKV_dict: dict) -> dict:
    # Extract the Q', K', V' probes
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    print("\nComputing the mutual information between the rows of Q_prime ...")
    P_Q_prime, MI_Q_prime, WSD_Q_prime = KDE_mutual_info_WSD(Q_prime, Q_prime, True)

    print("\nComputing the mutual information between the rows of K and K_prime ...")
    P_K_prime, MI_K_prime, WSD_K_prime = KDE_mutual_info_WSD(K_prime, K_prime, True)

    print("\nComputing the mutual information between the rows of V and V_prime ...")
    P_K_prime, MI_V_prime, WSD_K_prime = KDE_mutual_info_WSD(V_prime, V_prime, True)

    return dict(P_Q_prime=P_Q_prime, 
                MI_Q_prime=MI_Q_prime, 
                WSD_Q_prime=WSD_Q_prime,
                P_K_prime=P_K_prime,
                MI_K_prime=MI_K_prime,
                WSD_K_prime=WSD_K_prime,
                P_V_prime=P_K_prime,
                MI_V_prime=MI_V_prime,
                WSD_V_prime=WSD_K_prime)

def analyze_encoder_QKV_prime(analyzer) -> None:
    # Parameters for the analysis
    epoch = 19
    enc_layer = 0
    sentence_id = 3

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

    # Get the Q, K, V and Q', K', V' probes for the specific layer
    QKV_dict = analyzer.get_encoder_QKV(enc_layer, sentence_id, N_input_tokens)

    # Process the Q, K, V and Q', K', V' probes to obtain the MI and Wasserstein distances
    # between each probability row vector
    data_filename = Path(f"data/MI_WSD_dict_epoch_{epoch}_layer_{enc_layer}_sentence_{sentence_id}.pt")
    if data_filename.exists():
        print(f"Loading the MI and WSD data from {data_filename} ...")
        MI_WSD_dict = torch.load(data_filename, weights_only=False)
    else:
        MI_WSD_dict = process_QKV_prime_MI_WSD(QKV_dict)
        torch.save(MI_WSD_dict, data_filename)

    # Plot the Q, K, V and Q', K', V' mutual information and Wasserstein distance matrices
    plot_QKV_prime_MI_WSD(MI_WSD_dict, input_words, epoch, enc_layer, sentence_id)

    # Plot the PDF of Q', K', V' for specific words
    plot_QKV_prime_pdf(QKV_dict, MI_WSD_dict, input_words, epoch, enc_layer, sentence_id)

    # Process the Q, K, V and Q', K', V' matrices to obtain the softmax probabilities of
    # each row vector and the Wasserstein distance between the rows of the Q, K, V and 
    # Q', K', V' matrices
    softmax_div_dict = process_QKV_prime_softmax_distances(QKV_dict)

    # Plot the Jensen-Shannon divergence and the Bhattacharya coefficient between the rows of Q', K', V'
    plot_softmax_distances(softmax_div_dict, input_words, epoch, enc_layer, sentence_id)

    # Plot the PMF based on softmax probabilities for Q', K', V'
    plot_softmax_pmf(QKV_dict, softmax_div_dict, input_words, epoch, enc_layer, sentence_id)

    # Plot the Mutual Information and the Wasserstein distances as a scatter plot
    plot_MI_WSD_scatter(MI_WSD_dict, input_words, epoch, enc_layer, sentence_id)

    # Plot the Bhattachrya coefficient and Jensen-Shannon divergence as a scatter plot
    plot_BC_JSD_scatter(softmax_div_dict, input_words, epoch, enc_layer, sentence_id)
