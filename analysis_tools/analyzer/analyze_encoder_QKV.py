
import sys
sys.path.append('../../..')
sys.path.append('../../utils')
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import seaborn as sns
import pandas as pd


# Class implemented by this application
from mutual_info_estimator import MutualInfoEstimator
from compute_distances import compute_bhattacharya_coefficient, compute_wasserstein_distance
from compute_MI import compute_mutual_info, KDE_mutual_info

# ---------------------------------- Plotting routines start --------------------------------

def plot_QKV_and_QKV_prime_MI(MI_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int):
    # Extract the mutual information matrices
    MI_Q_Q_prime = MI_dict["MI_Q_Q_prime"]
    MI_K_K_prime = MI_dict["MI_K_K_prime"]
    MI_V_V_prime = MI_dict["MI_V_V_prime"]

    # Maximum threshold for the mutual information values
    max_val = 0.1

    # Flag to display the mutual information values on the plot
    display_values = False

    # Create the figure and axes
    fig, axs = plt.subplots(1,3)

    # Plot the QQ' mutual information matrix
    img = axs[0].imshow(MI_Q_Q_prime, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_Q_Q_prime.shape[0]):
            for j in range(MI_Q_Q_prime.shape[1]):
                text = axs[0].text(j, i, f"{MI_Q_Q_prime[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"MI between Q and Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the KK' mutual information matrix
    img = axs[1].imshow(MI_K_K_prime, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_K_K_prime.shape[0]):
            for j in range(MI_K_K_prime.shape[1]):
                text = axs[1].text(j, i, f"{MI_K_K_prime[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"MI between K and K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the VV' mutual information matrix
    img = axs[2].imshow(MI_V_V_prime, vmin=0, vmax=max_val, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(MI_V_V_prime.shape[0]):
            for j in range(MI_V_V_prime.shape[1]):
                text = axs[2].text(j, i, f"{MI_V_V_prime[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"MI between V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Mutual Information between Q, K, V and Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

def plot_softmax_distances(QKV_dict: dict, dist_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_probs = dist_dict["Q_probs"]
    Q_prime_probs = dist_dict["Q_prime_probs"]
    K_probs = dist_dict["K_probs"]
    K_prime_probs = dist_dict["K_prime_probs"]
    V_probs = dist_dict["V_probs"]
    V_prime_probs = dist_dict["V_prime_probs"]
    QQ_prime_wdist = dist_dict["QQ_prime_wdist"]
    KK_prime_wdist = dist_dict["KK_prime_wdist"]
    VV_prime_wdist = dist_dict["VV_prime_wdist"]
    QQ_prime_bhattacharya = dist_dict["QQ_prime_bhattacharya"]
    KK_prime_bhattacharya = dist_dict["KK_prime_bhattacharya"]
    VV_prime_bhattacharya = dist_dict["VV_prime_bhattacharya"]

    # Create the figure and axes
    # sns.set_theme(style="darkgrid", rc={'axes.facecolor':'lightcyan', 'figure.facecolor':'lightcyan'})
    sns.set_theme(style="darkgrid", rc={'axes.facecolor':'palegoldenrod', 'figure.facecolor':'white'})

    fig, axs = plt.subplots(4,3)

    # Word indices to plot
    word_indices=[4, 7, 10, 13]

    # ---------------------------------------------------------------------------------------------
    # Plot the probabilities of Q, K, V and Q', K', V' for specific words
    # ---------------------------------------------------------------------------------------------
    for i in range(4):
        P_Q = Q_probs[word_indices[i], :]
        P_Q_prime = Q_prime_probs[word_indices[i], :]
        dim_axis = np.arange(0, len(P_Q))

        #axs[a, 0].plot(dim_axis, P_Q, label='P_Q')
        #axs[a, 0].plot(dim_axis, P_Q_prime, label='P_Q_prime')
        sns.lineplot(x=dim_axis, y=P_Q, label='P(Q)', ax=axs[i, 0])
        sns.lineplot(x=dim_axis, y=P_Q_prime, label='P(Q_prime)', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"P(Q) and P(Q_prime) for the word '{input_words[word_indices[i]]}'")

        P_K = K_probs[word_indices[i], :]
        P_K_prime = K_prime_probs[word_indices[i], :]
        dim_axis = np.arange(0, len(P_K))

        # axs[a, 1].plot(dim_axis, P_K, label='P_K')
        # axs[a, 1].plot(dim_axis, P_K_prime, label='P_K_prime')
        sns.lineplot(x=dim_axis, y=P_K, label='P(K)', ax=axs[i, 1])
        sns.lineplot(x=dim_axis, y=P_K_prime, label='P(K_prime)', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"P(K) and P(K_prime) for the word '{input_words[word_indices[i]]}'")

        P_V = V_probs[word_indices[i], :]
        P_V_prime = V_prime_probs[word_indices[i], :]
        dim_axis = np.arange(0, len(P_V))

        # axs[a, 2].plot(dim_axis, P_V, label='P_V')
        # axs[a, 2].plot(dim_axis, P_V_prime, label='P_V_prime')
        sns.lineplot(x=dim_axis, y=P_V, label='P(V)', ax=axs[i, 2])
        sns.lineplot(x=dim_axis, y=P_V_prime, label='P(V_prime)', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"P(V) and P(V_prime) for the word '{input_words[word_indices[i]]}'")

    fig.suptitle(f"Softmax probabilities of Q, K, V and Q', K', V' for epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K, P(K) and K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(2, 1)
    K = QKV_dict["K"]
    K_prime = QKV_dict["K_prime"]
    P_K = K_probs[word_indices[1], :]
    P_K_prime = K_prime_probs[word_indices[1], :]

    k = K[word_indices[1]]
    # dataset = {'Vector Dimension': dim_axis, 'Coordinate Value': k}
    # df = pd.DataFrame(dataset)
    # sns.barplot(x='Vector Dimension', y='Coordinate Value', data=df, color='blue', ci=None, ax=axs[0])
    axs[0].bar(dim_axis, np.abs(k), color='blue', label='K')
    axs[0].set_xlabel('Vector Dimension')
    axs[0].set_ylabel('Coordinate Value')
    axs[0].set_title(f"K and P(K) for the word '{input_words[word_indices[1]]}'")
    axs[0].legend()
    ax_twin = axs[0].twinx()
    ax_twin.plot(dim_axis, P_K, color='red', label='P(K)')
    #sns.lineplot(x=dim_axis, y=P_K, color='red', ax=ax_twin)
    ax_twin.set_ylabel('Probability')

    k_prime = K_prime[word_indices[1]]
    # dataset = {'Vector Dimension': dim_axis, 'Coordinate Value': k_prime}
    # df = pd.DataFrame(dataset)
    # sns.barplot(x='Vector Dimension', y='Coordinate Value', data=df, color='blue', ci=None, ax=axs[1])
    axs[1].bar(dim_axis, np.abs(k_prime), color='blue', label='K\'')
    axs[1].set_xlabel('Vector Dimension')
    axs[1].set_ylabel('Coordinate Value')
    axs[1].set_title(f"K' and P(K') for the word '{input_words[word_indices[1]]}'")
    axs[1].legend()
    ax_twin = axs[1].twinx()
    #sns.lineplot(x=dim_axis, y=P_K_prime, color='red', ax=ax_twin)
    ax_twin.plot(dim_axis, P_K_prime, color='red', label='P_K\'')
    ax_twin.set_ylabel('Probability')

    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)


    # ---------------------------------------------------------------------------------------------
    # Plot the Wasserstein distance between the rows of Q, K, V and Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    QQ_prime_wdist = dist_dict["QQ_prime_wdist"]
    KK_prime_wdist = dist_dict["KK_prime_wdist"]
    VV_prime_wdist = dist_dict["VV_prime_wdist"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the QQ' Wasserstein distance matrix
    img = axs[0].imshow(QQ_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(QQ_prime_wdist.shape[0]):
            for j in range(QQ_prime_wdist.shape[1]):
                text = axs[0].text(j, i, f"{QQ_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Wasserstein distance between Q and Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the KK' Wasserstein distance matrix
    img = axs[1].imshow(KK_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(KK_prime_wdist.shape[0]):
            for j in range(KK_prime_wdist.shape[1]):
                text = axs[1].text(j, i, f"{KK_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Wasserstein distance between K and K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the VV' Wasserstein distance matrix
    img = axs[2].imshow(VV_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(VV_prime_wdist.shape[0]):
            for j in range(VV_prime_wdist.shape[1]):
                text = axs[2].text(j, i, f"{VV_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Wasserstein distance between V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Wasserstein distance between Q, K, V and Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Bhattacharya coefficient between the rows of Q, K, V and Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    QQ_prime_bhattacharya = dist_dict["QQ_prime_bhattacharya"]
    KK_prime_bhattacharya = dist_dict["KK_prime_bhattacharya"]
    VV_prime_bhattacharya = dist_dict["VV_prime_bhattacharya"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the QQ' Bhattacharya distance matrix
    img = axs[0].imshow(QQ_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(QQ_prime_bhattacharya.shape[0]):
            for j in range(QQ_prime_bhattacharya.shape[1]):
                text = axs[0].text(j, i, f"{QQ_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Bhattacharya coeff between Q and Q'")
    axs[0].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[0].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the KK' Bhattacharya distance matrix
    img = axs[1].imshow(KK_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(KK_prime_bhattacharya.shape[0]):
            for j in range(KK_prime_bhattacharya.shape[1]):
                text = axs[1].text(j, i, f"{KK_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[1].set_aspect('equal')
    axs[1].set_title(f"Bhattacharya coeff between K and K'")
    axs[1].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[1].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    # Plot the VV' Bhattacharya distance matrix
    img = axs[2].imshow(VV_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(VV_prime_bhattacharya.shape[0]):
            for j in range(VV_prime_bhattacharya.shape[1]):
                text = axs[2].text(j, i, f"{VV_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Bhattacharya coeff between V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Bhattacharya coeff between Q, K, V and Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)


def plot_KDE_distances(QKV_dict, dist_dict, input_words, epoch, enc_layer, sentence_id) -> None:
    Q_prime_probs = dist_dict["Q_prime_probs"]
    Q_probs = dist_dict["Q_probs"]
    Q_x_coord = dist_dict["Q_x_coord"]
    Q_prime_wdist = dist_dict["Q_prime_wdist"]
    Q_prime_bhattacharya = dist_dict["Q_prime_bhattacharya"]
    K_prime_probs = dist_dict["K_prime_probs"]
    K_probs = dist_dict["K_probs"]
    K_x_coord = dist_dict["K_x_coord"]
    K_prime_wdist = dist_dict["K_prime_wdist"]
    K_prime_bhattacharya = dist_dict["K_prime_bhattacharya"]
    V_prime_probs = dist_dict["V_prime_probs"]
    V_probs = dist_dict["V_probs"]
    V_x_coord = dist_dict["V_x_coord"]
    V_prime_wdist = dist_dict["V_prime_wdist"]
    V_prime_bhattacharya = dist_dict["V_prime_bhattacharya"]

    sns.set_theme(style="darkgrid", rc={'axes.facecolor':'palegoldenrod', 'figure.facecolor':'white'})

    # Word indices to plot
    word_indices=[4, 7, 10, 13]

    fig, axs = plt.subplots(4,3)

    # ---------------------------------------------------------------------------------------------
    # Plot the probabilities of Q', K', V' for specific words
    # ---------------------------------------------------------------------------------------------
    for i in range(4):
        row = word_indices[i]
        word_str = input_words[row]

        P_X = Q_prime_probs[row, :]
        P_Y = Q_probs[row, :]

        sns.lineplot(x=Q_x_coord, y=P_X, label='P(Q\')', ax=axs[i, 0])
        sns.lineplot(x=Q_x_coord, y=P_Y, label='P(Q)', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"Q and Q' probabilities for '{word_str}'")

        P_X = K_prime_probs[row, :]
        P_Y = K_probs[row, :]

        sns.lineplot(x=K_x_coord, y=P_X, label='P(K)', ax=axs[i, 1])
        sns.lineplot(x=K_x_coord, y=P_Y, label='P(K\')', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"K and K' probabilities for '{word_str}'")

        P_X = V_prime_probs[row, :]
        P_Y = V_probs[row, :]

        sns.lineplot(x=V_x_coord, y=P_X, label=f'P(V)', ax=axs[i, 2])
        sns.lineplot(x=V_x_coord, y=P_Y, label=f'P(V\')', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"V and V' probabilities for '{word_str}'")

    fig.suptitle(f"KDE probabilities of Q,K,V, and Q',K',V' for epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, ax = plt.subplots()
    row = word_indices[1]
    X_word = input_words[row]
    K_prime = QKV_dict["K_prime"]
    k_prime = K_prime[row]
    P_K_prime = K_prime_probs[row, :]

    # ax.bar(dim_axis, np.abs(q_prime), color='blue', label='K\'')
    N_bins = P_K_prime.shape[0]
    ax.hist(k_prime, bins=N_bins, color='blue', label='K\'')
    ax.set_xlabel('Coordinate Values')
    ax.set_ylabel('Counts')
    ax.set_title(f"K' and P(K') for the word '{X_word}'")
    ax.legend()
    ax_twin = ax.twinx()
    ax_twin.plot(K_x_coord, P_K_prime, color='red', label='P_K\'')
    ax_twin.set_ylabel('Probability')
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Wasserstein distance between the rows of Q, K, V and Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    Q_prime_wdist = dist_dict["Q_prime_wdist"]
    K_prime_wdist = dist_dict["K_prime_wdist"]
    V_prime_wdist = dist_dict["V_prime_wdist"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q' Wasserstein distance matrix
    img = axs[0].imshow(Q_prime_wdist, cmap=plt.cm.jet)
    if display_values:
        for i in range(Q_prime_wdist.shape[0]):
            for j in range(Q_prime_wdist.shape[1]):
                text = axs[0].text(j, i, f"{Q_prime_wdist[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Wasserstein distance between the rows of Q and Q'")
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
    axs[1].set_title(f"Wasserstein distance between the rows of K and K'")
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
    axs[2].set_title(f"Wasserstein distance between the rows of V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(img, cax=cax)
    fig.suptitle(f"Wasserstein distance between the rows of Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the Bhattacharya coefficient between the rows of Q, K, V and Q', K', V'
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, axs = plt.subplots(1,3)

    Q_prime_bhattacharya = dist_dict["Q_prime_bhattacharya"]
    K_prime_bhattacharya = dist_dict["K_prime_bhattacharya"]
    V_prime_bhattacharya = dist_dict["V_prime_bhattacharya"]

    # Flag to display the mutual information values on the plot
    display_values = True

    # Plot the Q' Bhattacharya distance matrix
    img = axs[0].imshow(Q_prime_bhattacharya, cmap=plt.cm.Wistia)
    if display_values:
        for i in range(Q_prime_bhattacharya.shape[0]):
            for j in range(Q_prime_bhattacharya.shape[1]):
                text = axs[0].text(j, i, f"{Q_prime_bhattacharya[i, j]:.2f}", horizontalalignment="center", verticalalignment="center", color="black", fontsize=6)
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Bhattacharya coeff between the rows of Q and Q'")
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
    axs[1].set_title(f"Bhattacharya coeff between the rows of K and K'")
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
    axs[2].set_title(f"Bhattacharya coeff between the rows of V and V'")
    axs[2].set_xticks(range(0, len(input_words)), input_words, rotation=90)
    axs[2].set_yticks(range(0, len(input_words)), input_words, rotation=0)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig.suptitle(f"Bhattacharya coeff between the rows of Q', K', V'\nfor epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)

# ---------------------------------- Plotting routines end ----------------------------------


def process_QKV_and_QKV_prime_softmax_distances(QKV_dict: dict) -> dict:
    # Extract the Q, K, V and Q', K', V' probes
    Q = QKV_dict["Q"]
    K = QKV_dict["K"]
    V = QKV_dict["V"]
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    # Parameters of the processing function
    N_rows = Q_prime.shape[0]
    beta = 5.0
    # PROB_THRESHOLD = 0.0015
    PROB_THRESHOLD = 0

    # Initialize the Q and Q' probabilty matrices
    Q_prime_probs = np.zeros(Q_prime.shape)
    Q_probs = np.zeros(Q.shape)

    for i in range(N_rows):
        X = Q_prime[i]
        X /= max(X)
        Y = Q[i]
        Y /= max(Y)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        P_Y = np.exp(beta * Y)/np.sum(np.exp(beta * Y))
        Q_prime_probs[i, :] = P_X
        Q_probs[i, :] = P_Y

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_Q_prime_probs = Q_prime_probs.copy()
    idx = np.where(Q_prime_probs < PROB_THRESHOLD)
    tmp_Q_prime_probs[idx] = 0

    tmp_Q_probs = Q_probs.copy()
    idx = np.where(Q_probs < PROB_THRESHOLD)
    tmp_Q_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the Q and Q' matrices
    QQ_prime_wdist = compute_wasserstein_distance(tmp_Q_prime_probs, tmp_Q_probs)

    # Compute the Bhattacharya coefficient between the rows of the Q and Q' matrices
    QQ_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_prime_probs, tmp_Q_probs)

    # Initialize the K and K' probabilty matrices
    K_prime_probs = np.zeros(K_prime.shape)
    K_probs = np.zeros(K.shape)

    for i in range(N_rows):
        X = np.abs(K_prime[i])
        X /= max(X)
        Y = np.abs(K[i])
        Y /= max(Y)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        P_Y = np.exp(beta * Y)/np.sum(np.exp(beta * Y))
        K_prime_probs[i, :] = P_X
        K_probs[i, :] = P_Y

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_K_prime_probs = K_prime_probs.copy()
    idx = np.where(K_prime_probs < PROB_THRESHOLD)
    tmp_K_prime_probs[idx] = 0

    tmp_K_probs = K_probs.copy()
    idx = np.where(K_probs < PROB_THRESHOLD)
    tmp_K_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the K and K' matrices
    KK_prime_wdist = compute_wasserstein_distance(tmp_K_prime_probs, tmp_K_probs)

    # Compute the Bhattacharya coefficient between the rows of the K and K' matrices
    KK_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_K_prime_probs, tmp_K_probs)

    # Initialize the V and V' probabilty matrices
    V_prime_probs = np.zeros(V_prime.shape)
    V_probs = np.zeros(V.shape)

    for i in range(N_rows):
        X = V_prime[i]
        X /= max(X)
        Y = V[i]
        Y /= max(Y)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        P_Y = np.exp(beta * Y)/np.sum(np.exp(beta * Y))
        V_prime_probs[i, :] = P_X
        V_probs[i, :] = P_Y

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_V_prime_probs = V_prime_probs.copy()
    idx = np.where(V_prime_probs < PROB_THRESHOLD)
    tmp_V_prime_probs[idx] = 0

    tmp_V_probs = V_probs.copy()
    idx = np.where(V_probs < PROB_THRESHOLD)
    tmp_V_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the V and V' matrices
    VV_prime_wdist = compute_wasserstein_distance(tmp_V_prime_probs, tmp_V_probs)

    # Compute the Bhattacharya coefficient between the rows of the V and V' matrices
    VV_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_V_prime_probs, tmp_V_probs)

    return dict(Q_probs=Q_probs, 
                Q_prime_probs=Q_prime_probs, 
                K_probs=K_probs, 
                K_prime_probs=K_prime_probs,
                V_probs=V_probs,
                V_prime_probs=V_prime_probs, 
                QQ_prime_wdist=QQ_prime_wdist, 
                KK_prime_wdist=KK_prime_wdist, 
                VV_prime_wdist=VV_prime_wdist,
                QQ_prime_bhattacharya=QQ_prime_bhattacharya,
                KK_prime_bhattacharya=KK_prime_bhattacharya,
                VV_prime_bhattacharya=VV_prime_bhattacharya)


def process_QKV_and_QKV_prime_KDE_distances(KDE_dict: dict) -> dict:
    # PROB_THRESHOLD = 0.0015
    PROB_THRESHOLD = 0

    # =================================================================================================
    # Process Q' KDE data
    # =================================================================================================
    Q_Q_prime_prob_matrix = KDE_dict["Q_Q_prime_prob_matrix"]
    N_rows = len(Q_Q_prime_prob_matrix)
    data = Q_Q_prime_prob_matrix[0][0]
    prob_data = data["prob_data"]
    P_Q_prime = prob_data["P_X"]
    P_Q = prob_data["P_Y"]
    Q_x_coord = prob_data["xpos"]

    Q_prime_probs = np.zeros((N_rows, P_Q_prime.shape[0]))
    for i in range(N_rows):
        data = Q_Q_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_Q_prime = prob_data["P_X"]
        Q_prime_probs[i, :] = P_Q_prime

    tmp_Q_prime_probs = Q_prime_probs.copy()
    idx = np.where(Q_prime_probs < PROB_THRESHOLD)
    tmp_Q_prime_probs[idx] = 0

    Q_probs = np.zeros((N_rows, P_Q.shape[0]))
    for i in range(N_rows):
        data = Q_Q_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_Q = prob_data["P_Y"]
        Q_probs[i, :] = P_Q

    tmp_Q_probs = Q_probs.copy()
    idx = np.where(Q_probs < PROB_THRESHOLD)
    tmp_Q_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the Q and Q' matrices
    Q_prime_wdist = compute_wasserstein_distance(tmp_Q_prime_probs, tmp_Q_probs)

    # Compute the Bhattacharya coefficient between the rows of the Q and Q' matrices
    Q_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_prime_probs, tmp_Q_probs)

    # =================================================================================================
    # Process K' KDE data
    # =================================================================================================
    K_K_prime_prob_matrix = KDE_dict["K_K_prime_prob_matrix"]
    N_rows = len(K_K_prime_prob_matrix)
    data = K_K_prime_prob_matrix[0][0]
    prob_data = data["prob_data"]
    P_K_prime = prob_data["P_X"]
    P_K = prob_data["P_Y"]
    K_x_coord = prob_data["xpos"]

    K_prime_probs = np.zeros((N_rows, P_K_prime.shape[0]))
    for i in range(N_rows):
        data = K_K_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_K_prime = prob_data["P_X"]
        K_prime_probs[i, :] = P_K_prime

    tmp_K_prime_probs = K_prime_probs.copy()
    idx = np.where(K_prime_probs < PROB_THRESHOLD)
    tmp_K_prime_probs[idx] = 0

    K_probs = np.zeros((N_rows, P_K.shape[0]))
    for i in range(N_rows):
        data = K_K_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_K = prob_data["P_Y"]
        K_probs[i, :] = P_K

    tmp_K_probs = K_probs.copy()
    idx = np.where(K_probs < PROB_THRESHOLD)
    tmp_K_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the K and K' matrices
    K_prime_wdist = compute_wasserstein_distance(tmp_K_prime_probs, tmp_K_probs)

    # Compute the Bhattacharya coefficient between the rows of the K and K' matrices
    K_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_K_prime_probs, tmp_K_probs)

    # =================================================================================================
    # Process V' KDE data
    # =================================================================================================
    V_V_prime_prob_matrix = KDE_dict["V_V_prime_prob_matrix"]
    N_rows = len(V_V_prime_prob_matrix)
    data = V_V_prime_prob_matrix[0][0]
    prob_data = data["prob_data"]
    P_V_prime = prob_data["P_X"]
    P_V = prob_data["P_Y"]
    V_x_coord = prob_data["xpos"]

    V_prime_probs = np.zeros((N_rows, P_V_prime.shape[0]))
    for i in range(N_rows):
        data = V_V_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_V_prime = prob_data["P_X"]
        V_prime_probs[i, :] = P_V_prime

    tmp_V_prime_probs = V_prime_probs.copy()
    idx = np.where(V_prime_probs < PROB_THRESHOLD)
    tmp_V_prime_probs[idx] = 0

    V_probs = np.zeros((N_rows, P_V.shape[0]))
    for i in range(N_rows):
        data = V_V_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_V = prob_data["P_Y"]
        V_probs[i, :] = P_V

    tmp_V_probs = V_probs.copy()
    idx = np.where(V_probs < PROB_THRESHOLD)
    tmp_V_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the V and V' matrices
    V_prime_wdist = compute_wasserstein_distance(tmp_V_prime_probs, tmp_V_probs)

    # Compute the Bhattacharya coefficient between the rows of the V and V' matrices
    V_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_V_prime_probs, tmp_V_probs)

    return dict(Q_prime_probs=Q_prime_probs, 
                Q_probs = Q_probs,
                Q_x_coord=Q_x_coord,
                Q_prime_wdist=Q_prime_wdist, 
                Q_prime_bhattacharya=Q_prime_bhattacharya,
                K_prime_probs=K_prime_probs,
                K_probs = K_probs,
                K_x_coord=K_x_coord,
                K_prime_wdist=K_prime_wdist,
                K_prime_bhattacharya=K_prime_bhattacharya,
                V_prime_probs=V_prime_probs,
                V_probs = V_probs,
                V_x_coord=V_x_coord,
                V_prime_wdist=V_prime_wdist,
                V_prime_bhattacharya=V_prime_bhattacharya)








def process_QKV_and_QKV_prime_MI(QKV_dict: dict) -> dict:
    # Extract the Q, K, V and Q', K', V' probes
    Q = QKV_dict["Q"]
    K = QKV_dict["K"]
    V = QKV_dict["V"]
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    print("\nComputing the mutual information between the rows of Q and Q_prime ...")
    MI_Q_Q_prime = compute_mutual_info(Q_prime, Q, False)

    print("\nComputing the mutual information between the rows of K and K_prime ...")
    MI_K_K_prime = compute_mutual_info(K_prime, K, False)

    print("\nComputing the mutual information between the rows of V and V_prime ...")
    MI_V_V_prime = compute_mutual_info(V_prime, V, False)

    return dict(MI_Q_Q_prime=MI_Q_Q_prime, MI_K_K_prime=MI_K_K_prime, MI_V_V_prime=MI_V_V_prime)


def process_QKV_and_QKV_prime_KDE_MI(QKV_dict: dict) -> dict:
    # Extract the Q, K, V and Q', K', V' probes
    Q = QKV_dict["Q"]
    K = QKV_dict["K"]
    V = QKV_dict["V"]
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    print("\nComputing the mutual information between the rows of Q_prime and Q ...")
    Q_Q_prime_prob_matrix, MI_Q_Q_prime = KDE_mutual_info(Q_prime, Q, False)

    print("\nComputing the mutual information between the rows of K_prime and K ...")
    K_K_prime_prob_matrix, MI_K_K_prime = KDE_mutual_info(K_prime, K, False)

    print("\nComputing the mutual information between the rows of V_prime and V ...")
    V_V_prime_prob_matrix, MI_V_V_prime = KDE_mutual_info(V_prime, V, False)

    return dict(Q_Q_prime_prob_matrix=Q_Q_prime_prob_matrix, 
                MI_Q_Q_prime=MI_Q_Q_prime,
                K_K_prime_prob_matrix=K_K_prime_prob_matrix,
                MI_K_K_prime=MI_K_K_prime,
                V_V_prime_prob_matrix=V_V_prime_prob_matrix,
                MI_V_V_prime=MI_V_V_prime)


def analyze_encoder_QKV(analyzer) -> None:
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

    # Get the words for the specific input sentence
    input_words = analyzer.get_encoder_input_words(sentence_id)

    # Get the Q, K, V and Q', K', V' probes for the specific layer
    QKV_dict = analyzer.get_encoder_QKV(enc_layer, sentence_id, N_input_tokens)

    # Process the Q, K, V and Q', K', V' probes
    MI_dict = process_QKV_and_QKV_prime_MI(QKV_dict)

    # Plot the Q, K, V and Q', K', V' mutual information matrices
    plot_QKV_and_QKV_prime_MI(MI_dict, input_words, epoch, enc_layer, sentence_id)

    # Process the Q, K, V and Q', K', V' matrices to obtain the softmax probabilities of
    # each row vector and the Wasserstein distance between the rows of the Q, K, V and 
    # Q', K', V' matrices
    dist_dict = process_QKV_and_QKV_prime_softmax_distances(QKV_dict)

    # Plot the wassertein distance between the rows of Q, K, V and Q', K', V'
    # and also the softmax probabilities of the rows of Q, K, V and Q', K', V'
    plot_softmax_distances(QKV_dict, dist_dict, input_words, epoch, enc_layer, sentence_id)

    # Special processing using KDE probabilities over the same range
    KDE_dict = process_QKV_and_QKV_prime_KDE_MI(QKV_dict)

    # Compute the wasserstein distance and bhattacharya coefficient using the KDE probabilities
    dist_dict = process_QKV_and_QKV_prime_KDE_distances(KDE_dict)

    # Plot the wassertein distance between the rows of Q, K, V and Q', K', V'
    # and also the softmax probabilities of the rows of Q, K, V and Q', K', V'
    plot_KDE_distances(QKV_dict, dist_dict, input_words, epoch, enc_layer, sentence_id)