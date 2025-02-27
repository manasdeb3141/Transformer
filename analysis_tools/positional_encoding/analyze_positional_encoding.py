import sys
sys.path.append('../utils')

import numpy as np

#import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# Definitions of the classes implemented by this application
from mutual_info_estimator import MutualInfoEstimator
from rotary_positional_encoding import rotary_position_encoding

def getPositionEncoding(seq_len, d, n=10000):
    PE = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            PE[k, 2*i] = np.sin(k/denominator)
            PE[k, 2*i+1] = np.cos(k/denominator)
    return PE
 

def plotSinusoid(t, d=512, n=10000):
    k = np.arange(0, 100, 1)
    denominator = np.power(n, 2*k/d)
    y = np.sin(t/denominator)
    plt.plot(k, y)
    plt.title('position = ' + str(t))
    plt.xlabel('dimension')
    if (t == 0):
        plt.ylabel('amplitude')
    plt.grid(True)

def main():
    # Create an example positional encoding matrix
    PE = getPositionEncoding(seq_len=4, d=4, n=100)
    print(f"Positional Encoding Matrix (seq_len=4, d_model=4, n=100):\n{PE}")

    # Plot sinusoids for different positions
    fig = plt.figure(figsize=(15, 4))    
    for i in range(4):
        plt.subplot(141 + i)
        plotSinusoid(i*4)
    plt.show()

    # Plot the heatmap of the PE matrix
    seq_len = 100
    d_model = 512
    PE = getPositionEncoding(seq_len=seq_len, d=d_model, n=10000)
    _, ax = plt.subplots()
    plt.set_cmap("jet") 
    cax = ax.matshow(PE)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xlabel('dimension')
    ax.set_ylabel('position')
    ax.set_title('Positional Encoding matrix (Seq length=100, d_model=512)')
    plt.gcf().colorbar(cax)
    plt.show()

    x = np.arange(0, 100, 1)
    y = np.arange(0, 100, 1)
    x_pos, y_pos = np.meshgrid(x, y)
    xy_pos = np.vstack([x_pos.ravel(), y_pos.ravel()]).T

    # Calculate MI for all pairs of positions
    print("Calculating MI for all pairs of positions...")
    MI = np.zeros((100, 100))
    for x, y in xy_pos:
        X = PE[x, :]
        Y = PE[y, :]
        MI_estimator = MutualInfoEstimator(X, Y)
        MI_data = MI_estimator.kraskov_MI()
        MI[x, y] = MI_data["MI"]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    surf = ax.plot_surface(x_pos, y_pos, MI, vmin=0, vmax=5, cmap=plt.cm.jet)
    fig.colorbar(surf, ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_zlabel('MI')

    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(MI, cmap=plt.cm.jet, origin='lower', vmin=0, vmax=5, extent=[0, MI.shape[0]-1, 0, MI.shape[1]-1])
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.grid(True)
    fig.colorbar(im, ax=ax)
    fig.suptitle('Mutual Information between Positional Encoding vectors\n(Seq length=100, d_model=512)')
    plt.show()

    # Generate the position encodings for the sequence
    # See https://karthick.ai/blog/2024/Rotatory-Position-Embedding-(RoPE)/
    PE_rotary = rotary_position_encoding(seq_len, d_model)
    print("Calculating MI for all pairs of rotary positions...")
    MI_rotary = np.zeros((100, 100))
    for x, y in xy_pos:
        X = PE_rotary[x, :]
        Y = PE_rotary[y, :]
        MI_estimator = MutualInfoEstimator(X, Y)
        MI_data = MI_estimator.kraskov_MI()
        MI_rotary[x, y] = MI_data["MI"]



    ax = fig.add_subplot(2, 1, 1, projection='3d')
    surf = ax.plot_surface(x_pos, y_pos, MI, vmin=0, vmax=5, cmap=plt.cm.jet)
    fig.colorbar(surf, ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_zlabel('MI')

    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(MI, cmap=plt.cm.jet, origin='lower', vmin=0, vmax=5, extent=[0, MI.shape[0]-1, 0, MI.shape[1]-1])
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.grid(True)
    fig.colorbar(im, ax=ax)
    fig.suptitle('Mutual Information between Rotary Positional Encoding vectors\n(Seq length=100, d_model=512)')
    plt.show()


if __name__ == "__main__":
    main()
