import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_dir = 'MNIST-Dataset/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY


def main():
    trX, trY, tsX, tsY = mnist(noTrSamples=400,
                               noTsSamples=100, digit_range=[5, 8],
                               noTrPerClass=200, noTsPerClass=50)


    trX = trX.T
    trY = trY.T
    tsX = tsX.T
    tsY = tsY.T

    #Performing PCA
    pca = PCA(n_components = 10)
    trX_reduced = pca.fit(trX).transform(trX)
    tsX_reduced = pca.fit(tsX).transform(tsX)
    covariance_trX_reduced = np.cov(trX_reduced.T)
    
    #Plotting covariance matrix
    plt.matshow(covariance_trX_reduced)
    plt.show()

    #Inverse transformation
    reconstructed_trX = pca.inverse_transform(trX_reduced)

    #Finding 5 samples from each class
    original_5 = []
    reconstructed_5 = []
    original_8 = []
    reconstructed_8 = []
    for i in range(len(trX)):
        if trY[i][0] == 5:
            if len(original_5) < 5:
                original_5.append(trX[i])
                reconstructed_5.append(reconstructed_trX[i])
        else:
            if len(original_8) < 5:
                original_8.append(trX[i])
                reconstructed_8.append(reconstructed_trX[i])


    #plotting 5 samples from each class: original image and inverse transformed image
    for i in range(5):
        nrows = 2
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        ax[0][0].imshow(original_5[i].reshape(28, -1))
        ax[0][0].set_title("Original 5")
        ax[0][1].imshow(reconstructed_5[i].reshape(28, -1))
        ax[0][1].set_title("Reconstructed 5")
        ax[1][0].imshow(original_8[i].reshape(28, -1))
        ax[1][0].set_title("Original 8")
        ax[1][1].imshow(reconstructed_8[i].reshape(28, -1))
        ax[1][1].set_title("Reconstructed 8")

        plt.tight_layout(True)
        plt.show()

    
if __name__ == "__main__":
    main()
