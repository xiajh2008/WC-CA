from sklearn import metrics
import WC_CA_IQR
import numpy as np

if __name__ == '__main__':
    path = 'Data sets/'
    file_name = 'Aggregation.txt'
    # file_name = 'CMC.txt'
    # file_name = 'Compound.txt'
    # file_name = 'D31.txt'
    # file_name = 'flame.txt'
    # file_name = 'jain.txt'
    # file_name = 'pathbased.txt'
    # file_name = 'R15.txt'
    # file_name = 'spiral.txt'

    data = np.loadtxt(path + file_name, delimiter=',')

    data_embedding = data[:, :-1]
    truth = data[:, -1]

    t = 9
    k = 11
    lamda = 0.7

    center = WC_CA_IQR.wc_ca_clustering(data_embedding, t, k, lamda)

    ari = metrics.adjusted_rand_score(center, truth)
    ami = metrics.adjusted_mutual_info_score(center, truth)
    nmi = metrics.normalized_mutual_info_score(center, truth)
    fmi = metrics.fowlkes_mallows_score(center, truth)

    print(ari, ami, nmi, fmi)