import numpy as np

class PCA:

    def __init__(self, data):
        self.data = data - np.average(data, axis = 0)
        self.U, self.S, self.V = np.linalg.svd(data, full_matrices= True)

    def transform(self, retained_variance = .9999):
        sReshaped = np.concatenate([np.diag(self.S), np.zeros((self.U.shape[0] - self.V.shape[0], self.V.shape[0]))], axis = 0)
        principalComponents = self.U @ sReshaped
        totalVar = np.sum(np.square(self.S))
        cumVar = np.cumsum(np.square(self.S))
        retainedVar = cumVar / totalVar
        for i in range(len(retainedVar)):
            if retainedVar[i] >= retained_variance:
                return principalComponents[:, :i + 1]