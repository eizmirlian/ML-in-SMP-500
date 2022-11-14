import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pca import PCA

class DBSCAN_Analysis:

    def __init__(self):
        data = pd.read_csv('S&P 500 Historical Data.csv')
        self.frame = pd.DataFrame(data)


        df = self.frame.copy()
        df = df.drop('Date', axis = 1)
        df = df.drop('Change %', axis = 1)
        df['Price'] = df['Price'].apply(lambda x: float(x.split()[0].replace(',', '')))
        df['Open'] = df['Open'].apply(lambda x: float(x.split()[0].replace(',', '')))
        df['High'] = df['High'].apply(lambda x: float(x.split()[0].replace(',', '')))
        df['Low'] = df['Low'].apply(lambda x: float(x.split()[0].replace(',', '')))
        scaling=StandardScaler()
        scaling.fit(df)
        self.scaledCopy = scaling.transform(df)

    def pcaTransform(self, retVariance):
        myPCA = PCA(self.scaledCopy)
        self.final = myPCA.transform(retained_variance = retVariance)

    def dbscan(self, testEps = .25, testMinSamples = 15):
        clusters = DBSCAN(eps = testEps, min_samples = testMinSamples).fit(self.final)

        self.labels = clusters.labels_
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise_ = list(self.labels).count(-1)

        print("Number of clusters: %d" % self.n_clusters_)
        print("Number of noise points: %d" % n_noise_)


    def evaluateClusters(self):
        N = self.labels.shape[0]
        clusterIncreaseNums = {}
        clusterSizes = {}
        for cluster in range(self.n_clusters_):
            clusterIncreaseNums[cluster] = 0
            clusterSizes[cluster] = 0
        for index in range(N):
            changeString = self.frame['Change %'][index]
            change = float(changeString[:-1])
            if self.labels[index] != -1:
                if change > 0:
                    clusterIncreaseNums[self.labels[index]] += 1
                clusterSizes[self.labels[index]] += 1
            clusterIncreaseRatios = {}
        for cluster in clusterSizes.keys():
            if clusterSizes[cluster] > 0:
                clusterIncreaseRatios[cluster] = clusterIncreaseNums[cluster] / clusterSizes[cluster]

        return self.clusterEntropies(clusterIncreaseRatios)

    def clusterEntropies(self, clusterRatios):
        clusterEntropies = {}
        totalEntropy = 0
        numClusters = len(clusterRatios.keys())
        for cluster in clusterRatios.keys():
            increaseProb = clusterRatios[cluster]
            decreaseProb = 1 - increaseProb
            entropy = -1 *(increaseProb * np.log2(increaseProb) + decreaseProb * np.log2(decreaseProb))
            clusterEntropies[cluster] = entropy
            totalEntropy += entropy
        if numClusters != 0:
            return clusterEntropies, totalEntropy / numClusters
        else:
            return None, 1
