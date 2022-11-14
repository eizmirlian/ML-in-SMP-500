import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dbscan import DBSCAN_Analysis

test = DBSCAN_Analysis()
test.pcaTransform(.999)
averageEntropies = []
testedVals = []
minEps = .01
maxEps = .3
epsVal = minEps
samplesVal = 3
while epsVal <= maxEps:
    testedVals.append(epsVal)
    test.dbscan(testEps = epsVal, testMinSamples = samplesVal)
    entropies, averageEntropy = test.evaluateClusters()
    averageEntropies.append(averageEntropy)
    epsVal += .01
    if (epsVal * 100) % 5 == 0:
        samplesVal += 3
scatter = plt.scatter(testedVals, averageEntropies)
plt.show()
