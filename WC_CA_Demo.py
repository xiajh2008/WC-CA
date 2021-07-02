from sklearn import metrics
import WC_CA
import numpy as np

if __name__ == '__main__':
    path = 'datasets/'
    # file_name = 'Aggregation.txt'
    # file_name = 'CMC.txt'
    # file_name = 'Compound.txt'
    # file_name = 'D31.txt'
    # file_name = 'flame.txt'
    # file_name = 'jain.txt'
    # file_name = 'pathbased.txt'
    # file_name = 'R15.txt'
    # file_name = 'spiral.txt'
    # file_name = 's2.txt'

    # file_name = 'ecoli.txt'
    # file_name = 'movement_libras.txt'
    # file_name = 'ionosphere.txt'
    file_name = 'iris.txt'
    # file_name = 'seeds.txt'
    # file_name = 'segmentation.txt'
    # file_name = 'wdbc.txt'
    # file_name = 'wine.txt'
    # file_name = 'spectrometer.txt'
    # file_name = 'glass.txt'
    # file_name = 'OlivettiFaces.txt'
    # file_name = 'usps.txt'
    # file_name = 'mnist.txt'


    data = np.loadtxt(path + file_name, delimiter=',')

    data_embedding = data[:, :-1]
    truth = data[:, -1]

    t = 8
    k = 27
    lamda = 0.9

    center = WC_CA.wc_ca_clustering(data_embedding, t, k, lamda)

    ari = metrics.adjusted_rand_score(center, truth)
    ami = metrics.adjusted_mutual_info_score(center, truth)
    fmi = metrics.fowlkes_mallows_score(center, truth)
    noise_ratio = len([x for x in center if x < 0]) / data_embedding.shape[0]
    print(f'ARI: {ari}, AMI: {ami}, FMI: {fmi}, noise_ratio: {noise_ratio}')