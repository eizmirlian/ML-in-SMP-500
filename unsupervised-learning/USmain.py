import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dbscan import DBSCAN_Analysis

test = DBSCAN_Analysis()
test.pcaTransform(.9999)
averageEntropies = []
testedVals = []
minEps = 0.0001
maxEps = .5
epsVal = minEps
samplesVal = 5
while epsVal <= maxEps:
    testedVals.append(epsVal)
    test.dbscan(testEps = epsVal, testMinSamples = samplesVal)
    entropies, averageEntropy = test.evaluateClusters()
    averageEntropies.append(averageEntropy)
    epsVal += .01
    #if (epsVal * 100) % 5 == 0:
        #samplesVal += 3
scatter = plt.scatter(testedVals, averageEntropies)
plt.xlabel('Eps Value')
plt.ylabel("Average Entropy per Cluster")
plt.show()
