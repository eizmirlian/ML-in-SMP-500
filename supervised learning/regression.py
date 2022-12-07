import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        diff = np.subtract(pred, label)
        rmse = np.sqrt(np.mean(np.square(diff)))
        return rmse

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        if x.ndim == 1:
            feat = []
            for row in x:
                new = []
                for deg in range(degree + 1):
                    new.append(np.power(row, deg))
                feat.append(new)
            return np.array(feat)
        for i in range(x.shape[0]):
            col1 = np.reshape(x[i, :], (x.shape[1], 1, 1))
            feat = np.concatenate((np.ones(col1.shape), col1), axis=1)
            for ii in range(2, degree + 1):
                feat = np.concatenate((feat, col1 ** ii), axis=1)
            if (i == 0):
                ret = feat
            else:
                ret = np.concatenate((ret, feat), axis=2)
        return np.transpose(ret)


    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return np.matmul(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:

        pinv = np.linalg.pinv(xtrain, rcond=1e-15)
        weights = np.matmul(pinv, ytrain)
        return weights

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]

        alpha = learning_rate
        weights = np.zeros([xtrain.shape[1], 1])
        loss_per_epoch = []
        pred = self.predict(xtrain, weights)
        for i in range(epochs):
            weights += (alpha/ytrain.shape[0]) * \
                np.matmul(xtrain.T, ytrain - pred)
            pred = self.predict(xtrain, weights)
            loss_per_epoch.append(self.rmse(ytrain, pred))
        return weights, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:

        data_num = xtrain.shape[0]
        weight = np.zeros([xtrain.shape[1], 1])
        loss_per_step = []
        for _ in range(epochs):
            for i in range(data_num):
                pred_i = self.predict(xtrain[i], weight)
                gradient = xtrain[i].T * (ytrain[i] - pred_i)
                gradient = np.reshape(gradient, (gradient.shape[0], 1))
                weight += gradient * learning_rate
                loss_per_step.append(
                    self.rmse(ytrain, self.predict(xtrain, weight)))
        return weight, loss_per_step

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:

        lamId = (np.eye(xtrain.shape[1]) * c_lambda)
        lamId[0][0] = 0
        print(lamId)
        inverse = np.linalg.pinv((xtrain.T @ xtrain + lamId))
        return inverse @ xtrain.T @ ytrain

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> float:  # [5 pts]
        print(X.shape)
        errors = []
        i = round(X.shape[0]/kfold)
        print(i)
        for k in range(kfold):
            ver_x = X[range(i*k, i*(k+1)), :]
            ver_y = y[range(i*k, i*(k+1)), :]
            train_x = np.concatenate(
                (X[0:i*k, :], X[i*(k+1):X.shape[0], :]), axis=0)
            train_y = np.concatenate(
                (y[0:i*k, :], y[i*(k+1):X.shape[0], :]), axis=0)
            print(kfold)
            print(train_x.shape)
            print('y')
            print(train_y)
            weight = self.ridge_fit_closed(train_x, train_y, c_lambda)
            errors.append(self.rmse(ver_y, self.predict(ver_x, weight)))
        return np.mean(np.array(errors))

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            error_list.append(err)
            if best_error is None or err < best_error:
                best_error = err
                best_lambda = lm

        return best_lambda, best_error, error_list
