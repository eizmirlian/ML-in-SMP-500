import numpy as np

class PCA:

    def __init__(self, data):
        self.data = data - np.average(data, axis = 0)
        self.U, self.S, self.V = np.linalg.svd(data, full_matrices= True)

    def transform(self, retained_variance = .99):
        principalComponents = self.U * self.S
        totalVar = np.sum(np.square(self.S))
        cumVar = np.cumsum(np.square(self.S))
        retainedVar = cumVar / totalVar
        for i in range(len(retainedVar)):
            if retainedVar[i] >= retained_variance:
                return principalComponents[:, :i + 1]