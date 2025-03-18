
import sys
sys.path.append('../../..')
sys.path.append('../../utils')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import seaborn as sns
from matplotlib import colors


# Class implemented by this application
from mutual_info_estimator import MutualInfoEstimator
from compute_distances import compute_bhattacharya_coefficient, compute_wasserstein_distance
from compute_MI import compute_mutual_info, KDE_mutual_info


# ---------------------------------- Plotting routines start --------------------------------

def plot_QKV_prime_MI(MI_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int):
    # Extract the mutual information matrices
    MI_Q_prime = MI_dict["MI_Q_prime"]
    MI_K_prime = MI_dict["MI_K_prime"]
    MI_V_prime = MI_dict["MI_V_prime"]

    # Fill the diagonal with zeros so that the plot is not dominated by the diagonal values
    # The diagonals are just entropies of the individual rows
    np.fill_diagonal(MI_Q_prime, 0)
    np.fill_diagonal(MI_K_prime, 0)
    np.fill_diagonal(MI_V_prime, 0)

    # Flag to display the mutual information values on the plot
    display_values = False

    # Create the figure and axes
    fig, axs = plt.subplots(1,3)

    # Plot the QQ' mutual information matrix
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

    # Plot the KK' mutual information matrix
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

    # Plot the VV' mutual information matrix
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

def plot_softmax_distances(QKV_dict: dict, dist_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_prime_probs = dist_dict["Q_prime_probs"]
    K_prime_probs = dist_dict["K_prime_probs"]
    V_prime_probs = dist_dict["V_prime_probs"]
    Q_prime_wdist = dist_dict["Q_prime_wdist"]
    K_prime_wdist = dist_dict["K_prime_wdist"]
    V_prime_wdist = dist_dict["V_prime_wdist"]
    Q_prime_bhattacharya = dist_dict["Q_prime_bhattacharya"]
    K_prime_bhattacharya = dist_dict["K_prime_bhattacharya"]
    V_prime_bhattacharya = dist_dict["V_prime_bhattacharya"]

    # Create the figure and axes
    # sns.set_theme(style="darkgrid", rc={'axes.facecolor':'lightcyan', 'figure.facecolor':'lightcyan'})
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

        P_X = Q_prime_probs[row_tuple[0], :]
        P_Y = Q_prime_probs[row_tuple[1], :]
        dim_axis = np.arange(0, len(P_X))

        sns.lineplot(x=dim_axis, y=P_X, label=f'P({X_word})', ax=axs[i, 0])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P({Y_word})', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"Q' probabilities")

        P_X = K_prime_probs[row_tuple[0], :]
        P_Y = K_prime_probs[row_tuple[1], :]
        dim_axis = np.arange(0, len(P_X))

        sns.lineplot(x=dim_axis, y=P_X, label=f'P({X_word})', ax=axs[i, 1])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P({Y_word})', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"K' probabilities")

        P_X = V_prime_probs[row_tuple[0], :]
        P_Y = V_prime_probs[row_tuple[1], :]
        dim_axis = np.arange(0, len(P_X))

        sns.lineplot(x=dim_axis, y=P_X, label=f'P({X_word})', ax=axs[i, 2])
        sns.lineplot(x=dim_axis, y=P_Y, label=f'P({Y_word})', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"V' probabilities")


    fig.suptitle(f"Softmax probabilities of Q', K', V' for epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    row_tuple = word_indices[1]
    X_word = input_words[row_tuple[0]]
    Y_word = input_words[row_tuple[1]]
    K_prime = QKV_dict["K_prime"]
    k_prime = K_prime[row_tuple[0]]
    P_K_prime = K_prime_probs[row_tuple[0], :]

    ax.bar(dim_axis, np.abs(k_prime), color='blue', label='K\'')
    ax.set_xlabel('Vector Dimension')
    ax.set_ylabel('Coordinate Value')
    ax.set_title(f"K' and P(K') for the word '{X_word}'")
    ax.legend()
    ax_twin = ax.twinx()
    ax_twin.plot(dim_axis, P_K_prime, color='red', label='P_K\'')
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


def plot_KDE_distances(QKV_dict, dist_dict, input_words, epoch, enc_layer, sentence_id) -> None:
    Q_prime_probs = dist_dict["Q_prime_probs"]
    Q_x_coord = dist_dict["Q_x_coord"]
    Q_prime_wdist = dist_dict["Q_prime_wdist"]
    Q_prime_bhattacharya = dist_dict["Q_prime_bhattacharya"]
    K_prime_probs = dist_dict["K_prime_probs"]
    K_x_coord = dist_dict["K_x_coord"]
    K_prime_wdist = dist_dict["K_prime_wdist"]
    K_prime_bhattacharya = dist_dict["K_prime_bhattacharya"]
    V_prime_probs = dist_dict["V_prime_probs"]
    V_x_coord = dist_dict["V_x_coord"]
    V_prime_wdist = dist_dict["V_prime_wdist"]
    V_prime_bhattacharya = dist_dict["V_prime_bhattacharya"]

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

        P_X = Q_prime_probs[row_tuple[0], :]
        P_Y = Q_prime_probs[row_tuple[1], :]

        sns.lineplot(x=Q_x_coord, y=P_X, label=f'P({X_word})', ax=axs[i, 0])
        sns.lineplot(x=Q_x_coord, y=P_Y, label=f'P({Y_word})', ax=axs[i, 0])
        axs[i, 0].legend()
        axs[i, 0].set_xlabel('Vector Dimension')
        axs[i, 0].set_ylabel('Probability')
        axs[i, 0].set_title(f"Q' probabilities")

        P_X = K_prime_probs[row_tuple[0], :]
        P_Y = K_prime_probs[row_tuple[1], :]

        sns.lineplot(x=K_x_coord, y=P_X, label=f'P({X_word})', ax=axs[i, 1])
        sns.lineplot(x=K_x_coord, y=P_Y, label=f'P({Y_word})', ax=axs[i, 1])
        axs[i, 1].legend()
        axs[i, 1].set_xlabel('Vector Dimension')
        axs[i, 1].set_ylabel('Probability')
        axs[i, 1].set_title(f"K' probabilities")

        P_X = V_prime_probs[row_tuple[0], :]
        P_Y = V_prime_probs[row_tuple[1], :]

        sns.lineplot(x=V_x_coord, y=P_X, label=f'P({X_word})', ax=axs[i, 2])
        sns.lineplot(x=V_x_coord, y=P_Y, label=f'P({Y_word})', ax=axs[i, 2])
        axs[i, 2].legend()
        axs[i, 2].set_xlabel('Vector Dimension')
        axs[i, 2].set_ylabel('Probability')
        axs[i, 2].set_title(f"V' probabilities")

    fig.suptitle(f"Softmax probabilities of Q', K', V' for epoch {epoch}, encoder layer {enc_layer}, sentence {sentence_id}")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=True)

    # ---------------------------------------------------------------------------------------------
    # Plot the K', P(K') for a specific word
    # ---------------------------------------------------------------------------------------------
    sns.reset_orig()
    plt.style.context('classic')
    fig, ax = plt.subplots()
    row_tuple = word_indices[1]
    X_word = input_words[row_tuple[0]]
    Y_word = input_words[row_tuple[1]]
    K_prime = QKV_dict["K_prime"]
    k_prime = K_prime[row_tuple[0]]
    P_K_prime = K_prime_probs[row_tuple[0], :]

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

def plot_BW_scatter(dist_dict: dict, input_words: list, epoch: int, enc_layer: int, sentence_id: int) -> None:
    Q_prime_bhattacharya = dist_dict["Q_prime_bhattacharya"]
    K_prime_bhattacharya = dist_dict["K_prime_bhattacharya"]
    V_prime_bhattacharya = dist_dict["V_prime_bhattacharya"]
    Q_prime_wdist = dist_dict["Q_prime_wdist"]
    K_prime_wdist = dist_dict["K_prime_wdist"]
    V_prime_wdist = dist_dict["V_prime_wdist"]

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


    for idx, word_idx in enumerate(word_indices):
        sns.set_theme(style="darkgrid")
        fig, axs = plt.subplots(1,3)
        word_str = input_words[word_idx]

        # Plot Q' Bhattacharya vs Q' Wasserstein distance
        x = Q_prime_bhattacharya[word_idx, :]
        y = Q_prime_wdist[word_idx, :]
        axs[0].scatter(x, y, c=word_colors)
        axs[0].set_xlabel('Bhattacharya Coefficient')
        axs[0].set_ylabel('Wasserstein Distance')
        axs[0].set_title(f"Q' probability space for the word '{word_str}'")
        # axs[0].grid(True)
        for i, txt in enumerate(input_words):
            axs[0].annotate(txt, (x[i], y[i]))

        # Plot K' Bhattacharya vs K' Wasserstein distance
        x = K_prime_bhattacharya[word_idx, :]
        y = K_prime_wdist[word_idx, :]
        axs[1].scatter(x, y, c=word_colors)
        axs[1].set_xlabel('Bhattacharya Coefficient')
        axs[1].set_ylabel('Wasserstein Distance')
        axs[1].set_title(f"K' probability space for the word '{word_str}'")
        # axs[1].grid(True)
        for i, txt in enumerate(input_words):
            axs[1].annotate(txt, (x[i], y[i]))

        # Plot V' Bhattacharya vs V' Wasserstein distance
        x = V_prime_bhattacharya[word_idx, :]
        y = V_prime_wdist[word_idx, :]
        axs[2].scatter(x, y, c=word_colors)
        axs[2].set_xlabel('Bhattacharya Coefficient')
        axs[2].set_ylabel('Wasserstein Distance')
        axs[2].set_title(f"V' probability space for the word '{word_str}'")
        # axs[2].grid(True)
        for i, txt in enumerate(input_words):
            axs[2].annotate(txt, (x[i], y[i]))

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
    beta = 5.0
    # PROB_THRESHOLD = 0.0015
    PROB_THRESHOLD = 0

    # Initialize the Q' probabilty matrix
    Q_prime_probs = np.zeros(Q_prime.shape)

    for i in range(N_rows):
        X = Q_prime[i]
        X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        Q_prime_probs[i, :] = P_X

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_Q_prime_probs = Q_prime_probs.copy()
    idx = np.where(Q_prime_probs < PROB_THRESHOLD)
    tmp_Q_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the Q' matrices
    Q_prime_wdist = compute_wasserstein_distance(tmp_Q_prime_probs, tmp_Q_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the Q' matrices
    Q_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_Q_prime_probs, tmp_Q_prime_probs)

    # Initialize the K and K' probabilty matrices
    K_prime_probs = np.zeros(K_prime.shape)

    for i in range(N_rows):
        X = np.abs(K_prime[i])
        X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        K_prime_probs[i, :] = P_X

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_K_prime_probs = K_prime_probs.copy()
    idx = np.where(K_prime_probs < PROB_THRESHOLD)
    tmp_K_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the K' matrices
    K_prime_wdist = compute_wasserstein_distance(tmp_K_prime_probs, tmp_K_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the K' matrices
    K_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_K_prime_probs, tmp_K_prime_probs)

    # Initialize the V and V' probabilty matrices
    V_prime_probs = np.zeros(V_prime.shape)

    for i in range(N_rows):
        X = V_prime[i]
        X /= max(X)
        P_X = np.exp(beta * X)/np.sum(np.exp(beta * X))
        V_prime_probs[i, :] = P_X

    # Threshold the probabilities before calculating the
    # Wasserstein distance and Bhattacharya coefficient
    tmp_V_prime_probs = V_prime_probs.copy()
    idx = np.where(V_prime_probs < PROB_THRESHOLD)
    tmp_V_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the V' matrices
    V_prime_wdist = compute_wasserstein_distance(tmp_V_prime_probs, tmp_V_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the V' matrices
    V_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_V_prime_probs, tmp_V_prime_probs)

    return dict(Q_prime_probs=Q_prime_probs, 
                K_prime_probs=K_prime_probs,
                V_prime_probs=V_prime_probs, 
                Q_prime_wdist=Q_prime_wdist, 
                K_prime_wdist=K_prime_wdist, 
                V_prime_wdist=V_prime_wdist,
                Q_prime_bhattacharya=Q_prime_bhattacharya,
                K_prime_bhattacharya=K_prime_bhattacharya,
                V_prime_bhattacharya=V_prime_bhattacharya)

def process_QKV_prime_KDE_distances(KDE_dict: dict) -> dict:
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
    Q_x_coord = prob_data["xpos"]

    Q_prime_probs = np.zeros((N_rows, P_Q.shape[0]))
    for i in range(N_rows):
        data = Q_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_Q = prob_data["P_X"]
        Q_prime_probs[i, :] = P_Q

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
    K_x_coord = prob_data["xpos"]

    K_prime_probs = np.zeros((N_rows, P_K.shape[0]))
    for i in range(N_rows):
        data = K_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_K = prob_data["P_X"]
        K_prime_probs[i, :] = P_K

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
    V_x_coord = prob_data["xpos"]

    V_prime_probs = np.zeros((N_rows, P_V.shape[0]))
    for i in range(N_rows):
        data = V_prime_prob_matrix[i][0]
        prob_data = data["prob_data"]
        P_V = prob_data["P_X"]
        V_prime_probs[i, :] = P_V

    tmp_V_prime_probs = V_prime_probs.copy()
    idx = np.where(V_prime_probs < PROB_THRESHOLD)
    tmp_V_prime_probs[idx] = 0

    # Compute the Wasserstein distance between the rows of the V' matrices
    V_prime_wdist = compute_wasserstein_distance(tmp_V_prime_probs, tmp_V_prime_probs)

    # Compute the Bhattacharya coefficient between the rows of the V' matrices
    V_prime_bhattacharya = compute_bhattacharya_coefficient(tmp_V_prime_probs, tmp_V_prime_probs)

    return dict(Q_prime_probs=Q_prime_probs, 
                Q_x_coord=Q_x_coord,
                Q_prime_wdist=Q_prime_wdist, 
                Q_prime_bhattacharya=Q_prime_bhattacharya,
                K_prime_probs=K_prime_probs,
                K_x_coord=K_x_coord,
                K_prime_wdist=K_prime_wdist,
                K_prime_bhattacharya=K_prime_bhattacharya,
                V_prime_probs=V_prime_probs,
                V_x_coord=V_x_coord,
                V_prime_wdist=V_prime_wdist,
                V_prime_bhattacharya=V_prime_bhattacharya)


def process_QKV_prime_MI(QKV_dict: dict) -> dict:
    # Extract the Q', K', V' probes
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    print("\nComputing the mutual information between the rows of Q_prime ...")
    MI_Q_prime = compute_mutual_info(Q_prime, Q_prime, True)

    print("\nComputing the mutual information between the rows of K and K_prime ...")
    MI_K_prime = compute_mutual_info(K_prime, K_prime, True)

    print("\nComputing the mutual information between the rows of V and V_prime ...")
    MI_V_prime = compute_mutual_info(V_prime, V_prime, True)

    return dict(MI_Q_prime=MI_Q_prime, MI_K_prime=MI_K_prime, MI_V_prime=MI_V_prime)

def process_QKV_prime_KDE_MI(QKV_dict: dict) -> dict:
    # Extract the Q', K', V' probes
    Q_prime = QKV_dict["Q_prime"]
    K_prime = QKV_dict["K_prime"]
    V_prime = QKV_dict["V_prime"]

    print("\nComputing the mutual information between the rows of Q_prime ...")
    Q_prime_prob_matrix, MI_Q_prime = KDE_mutual_info(Q_prime, Q_prime, True)

    print("\nComputing the mutual information between the rows of K_prime ...")
    K_prime_prob_matrix, MI_K_prime = KDE_mutual_info(K_prime, K_prime, True)

    print("\nComputing the mutual information between the rows of V_prime ...")
    V_prime_prob_matrix, MI_V_prime = KDE_mutual_info(V_prime, V_prime, True)

    return dict(Q_prime_prob_matrix=Q_prime_prob_matrix, 
                MI_Q_prime=MI_Q_prime,
                K_prime_prob_matrix=K_prime_prob_matrix,
                MI_K_prime=MI_K_prime,
                V_prime_prob_matrix=V_prime_prob_matrix,
                MI_V_prime=MI_V_prime)

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

    # Process the Q, K, V and Q', K', V' probes
    MI_dict = process_QKV_prime_MI(QKV_dict)

    # Plot the Q, K, V and Q', K', V' mutual information matrices
    plot_QKV_prime_MI(MI_dict, input_words, epoch, enc_layer, sentence_id)

    # Process the Q, K, V and Q', K', V' matrices to obtain the softmax probabilities of
    # each row vector and the Wasserstein distance between the rows of the Q, K, V and 
    # Q', K', V' matrices
    dist_dict = process_QKV_prime_softmax_distances(QKV_dict)

    # Plot the wassertein distance between the rows of Q', K', V'
    # and also the softmax probabilities of the rows of Q', K', V'
    plot_softmax_distances(QKV_dict, dist_dict, input_words, epoch, enc_layer, sentence_id)

    # Plot the bhattacharya coeff and wasserstein distances as a scatter plot
    plot_BW_scatter(dist_dict, input_words, epoch, enc_layer, sentence_id)

    # Special processing using KDE probabilities over the same range
    KDE_dict = process_QKV_prime_KDE_MI(QKV_dict)

    # Compute the wasserstein distance and bhattacharya coefficient using the KDE probabilities
    dist_dict = process_QKV_prime_KDE_distances(KDE_dict)

    # Plot the wassertein distance between the rows of Q', K', V'
    # and also the softmax probabilities of the rows of Q', K', V'
    plot_KDE_distances(QKV_dict, dist_dict, input_words, epoch, enc_layer, sentence_id)

    # Plot the KDE based bhattacharya coeff and wasserstein distances as a scatter plot
    plot_BW_scatter(dist_dict, input_words, epoch, enc_layer, sentence_id)