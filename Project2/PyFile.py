# # from scipy import optimize
# # import numpy as np
# # from cvxopt import matrix as cvxopt_matrix
# # from cvxopt import solvers as cvxopt_solvers


# # #New dataset (for later)
# # X = np.array([[3,4],[1,4],[2,3],[6,-1],[7,-1],[5,-3],[2,4]] )
# # y = np.array([-1,-1, -1, 1, 1 , 1, 1 ])

# # C = 10
# # m,n = X.shape
# # y = y.reshape(-1,1) * 1.
# # X_dash = y * X
# # H = np.dot(X_dash , X_dash.T) * 1.

# # #Converting into cvxopt format - as previously
# # P = cvxopt_matrix(H)
# # q = cvxopt_matrix(-np.ones((m, 1)))
# # G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
# # h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
# # A = cvxopt_matrix(y.reshape(1, -1))
# # b = cvxopt_matrix(np.zeros(1))

# # #Run solver
# # sol = cvxopt_solvers.qp(P, q, G, h, A, b)
# # alphas = np.array(sol['x'])

# # #==================Computing and printing parameters===============================#
# # w = ((y * alphas).T @ X).reshape(-1,1)
# # S = (alphas > 1e-4).flatten()
# # b = y[S] - np.dot(X[S], w)

# # #Display results
# # print('Alphas = ',alphas[alphas > 1e-4])
# # print('w = ', w.flatten())
# # print('b = ', b[0])

# from scipy import optimize
# import numpy as np
# from cvxopt import matrix as cvxopt_matrix
# from cvxopt import solvers as cvxopt_solvers
# import csv
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import svm

# df = pd.read_csv('pulsar_star_dataset.csv')
# X = df.drop('Class', axis=1)
# y = df['Class']
# X = X.to_numpy()
# y = y.to_numpy()
# y[y == 0] = -1
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# mean_train = X_train.mean()
# std_train = X_train.std()
# X_train_normalized = (X_train - mean_train) / std_train
# X_test_normalized = (X_test - mean_train) / std_train

# # clf = svm.SVC(kernel='linear', C=1)
# # clf.fit(X_train_normalized, y_train)
# # X_test_normalized = (X_test - mean_train) / std_train
# # y_pred = clf.predict(X_test_normalized)
# # a = accuracy_score(y_test, y_pred)
# # print(a)
# # #print alphas
# # #print(w)
# # #print(b)

# # print('w = ',clf.coef_)
# # print('b = ',clf.intercept_)
# # print('Indices of support vectors = ', clf.support_)
# # print('Support vectors = ', clf.support_vectors_)
# # print('Number of support vectors for each class = ', clf.n_support_)
# # print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
# C = 1.0
# m,n = X_train_normalized.shape
# y_train = y_train.reshape(-1,1) * 1.
# X_dash = y_train * X_train_normalized
# H = np.dot(X_dash , X_dash.T) * 1.

# #Converting into cvxopt format - as previously
# P = cvxopt_matrix(H)
# q = cvxopt_matrix(-np.ones((m, 1)))
# G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
# h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
# A = cvxopt_matrix(y_train.reshape(1, -1))
# b = cvxopt_matrix(np.zeros(1))

# #Run solver
# sol = cvxopt_solvers.qp(P, q, G, h, A, b)
# alphas = np.array(sol['x'])

# #==================Computing and printing parameters===============================#
# w = ((y * alphas).T @ X).reshape(-1,1)
# S = (alphas > 1e-4).flatten()
# b = y[S] - np.dot(X[S], w)

# #Display results
# print('Alphas = ',alphas[alphas > 1e-4])
# print('w = ', w.flatten())
# print('b = ', b[0])

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

# def gaussian_kernel(x, y):
#     return np.exp(-0.1*linalg.norm(x-y)**2)

def gaussian_kernel(x, y):                                                     # for matrices
        X_norm = np.sum(x ** 2, axis = -1)
        Y_norm = np.sum(y.T ** 2, axis = -1)
        return np.exp(-0.1*(X_norm[:,None] + Y_norm[None,:] - 2 * np.dot(x, y)))

# class SVM(object):

#     def __init__(self, kernel=linear_kernel, C=None):
#         self.kernel = kernel
#         self.C = C
#         if self.C is not None: self.C = float(self.C)

#     def fit(self, X, y):
#         n_samples, n_features = X.shape

#         # Gram matrix
#         K = np.zeros((n_samples, n_samples))
#         # for i in range(n_samples):
#         #     for j in range(n_samples):
#         #         K[i,j] = self.kernel(X[i], X[j])
#         K = self.kernel(X, X.T)

#         P = cvxopt.matrix(np.outer(y,y) * K)
#         q = cvxopt.matrix(np.ones(n_samples) * -1)
#         A = cvxopt.matrix(y, (1,n_samples)) * 1.
#         b = cvxopt.matrix(0.0) * 1.

#         if self.C is None:
#             G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
#             h = cvxopt.matrix(np.zeros(n_samples))
#         else:
#             tmp1 = np.diag(np.ones(n_samples) * -1)
#             tmp2 = np.identity(n_samples)
#             G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
#             tmp1 = np.zeros(n_samples)
#             tmp2 = np.ones(n_samples) * self.C
#             h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

#         # solve QP problem
#         solution = cvxopt.solvers.qp(P, q, G, h, A, b)

#         # Lagrange multipliers
#         a = np.ravel(solution['x'])

#         # Support vectors have non zero lagrange multipliers
#         sv = a > 1e-5
#         ind = np.arange(len(a))[sv]
#         self.a = a[sv]
#         self.sv = X[sv]
#         self.sv_y = y[sv]
#         print("%d support vectors out of %d points" % (len(self.a), n_samples))

#         # Intercept
#         self.b = 0
#         for n in range(len(self.a)):
#             self.b += self.sv_y[n]
#             self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
#         self.b /= len(self.a)

#         # Weight vector
#         if self.kernel == linear_kernel:
#             self.w = np.zeros(n_features)
#             for n in range(len(self.a)):
#                 self.w += self.a[n] * self.sv_y[n] * self.sv[n]
#         else:
#             self.w = None

#     def project(self, X):
#         if self.w is not None:
#             return np.dot(X, self.w) + self.b
#         else:
#             y_predict = np.zeros(len(X))
#             # for i in range(len(X)):
#             #     s = 0
#             #     for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
#             #         if self.kernel == polynomial_kernel:
#             #             s += a * sv_y * self.kernel(X[i], sv)
#             #         else:
#             #             s += a * sv_y * self.kernel(X[i], sv)
#             #     y_predict[i] = s
#             y_predict = np.sum(self.a * self.sv_y * self.kernel(X, self.sv.T), axis=1)
#             return y_predict + self.b

#     def predict(self, X):
#         return np.sign(self.project(X))

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
             

class SVM(object):

    def linear_kernel(self, x1, x2):                                                            # defining the kernel functions 
        return np.dot(x1, x2) * 1.                                                            # multiplying here and elsewhere by 1. to convert to float

    def quadratic_kernel(self, x1, x2):
        return ((1 + np.dot(x1, x2))*1.) ** 2                                

    def gaussian_kernel(self, x1, x2):                                                     # for matrices
        X_norm = np.sum(x1 ** 2, axis = -1)
        Y_norm = np.sum(x2 ** 2, axis = -1)
        return np.exp(-self.gamma * (X_norm[:,None] + Y_norm[None,:] - 2 * np.dot(x1, x2)))

    def __init__(self, kernel_str='linear', C=None, gamma=0.1):                                 # initializing the SVM class
        if kernel_str == 'linear':
            self.kernel = SVM.linear_kernel
        elif kernel_str == 'quadratic':
            self.kernel = SVM.quadratic_kernel
        elif kernel_str == 'gaussian':
            self.kernel = SVM.gaussian_kernel
        else:
            self.kernel = SVM.linear_kernel
            print('Invalid kernel string, defaulting to linear.')
        self.C = C
        self.gamma = gamma
        self.kernel_str = kernel_str
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        kernel_matrix = np.zeros((num_samples, num_samples))                                                    # creating the kernel matrix
        kernel_matrix = self.kernel(self, X, X.T)

        P = cvxopt.matrix(np.outer(y,y) * kernel_matrix)                                                    # creating the matrices for the dual optimization problem, derivation explained in report
        q = cvxopt.matrix(np.ones(num_samples) * -1)
        A = cvxopt.matrix(y, (1,num_samples)) * 1.
        b = cvxopt.matrix(0) * 1.

        if self.C is None:                                                                                 # if C is not specified, then the problem is hard margin
            G = cvxopt.matrix(np.diag(np.ones(num_samples) * -1))
            h = cvxopt.matrix(np.zeros(num_samples))
        else:                                                                                              # if C is specified, then the problem is soft margin
            tmp1 = np.diag(np.ones(num_samples) * -1)
            tmp2 = np.identity(num_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(num_samples)
            tmp2 = np.ones(num_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))


        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        
        a = np.ravel(solution['x'])                                                                         # get the lagrange multipliers from the solution
        support_vectors = a > 1e-5                                                                          # get the support vectors which have non-zero lagrange multipliers
        ind = np.arange(len(a))[support_vectors]                                                            # get the indices of the support vectors for the kernel matrix
        self.a = a[support_vectors]                                                                         # storing the data of the solution in the svm object
        self.support_vectors = X[support_vectors]
        self.y_support_vectors = y[support_vectors]
        print("%d support vectors out of %d points" % (len(self.a), num_samples))

        self.b = 0                                                                                          # deriving the bias value by enforcing the condition for b in the svm optimization problem
        for n in range(len(self.a)):
            self.b += self.y_support_vectors[n]
            self.b -= np.sum(self.a * self.y_support_vectors * kernel_matrix[ind[n],support_vectors])
        self.b /= len(self.a)

        if self.kernel_str == 'linear':                                                                     # deriving the weights for the linear kernel
            self.w = np.zeros(num_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.y_support_vectors[n] * self.support_vectors[n]
        else:
            self.w = None                                                                                   # if the kernel is not linear, then the weights are not defined

    def predict(self, X):
        return np.sign(self.project(X))                                                                     # predicting the class of the input data by taking the sign of the projection
        
    def project(self, X):
        if self.kernel_str == 'linear':                                                                     # if linear, then the projection is given by the linear combination of the support vectors
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.sum(self.a * self.y_support_vectors * self.kernel(self, X, self.support_vectors.T), axis=1)
            y_predict = y_predict + self.b                                                                  # if not linear, then the projection is given by the kernel trick
            return y_predict
            


if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.support_vectors[:,0], clf.support_vectors[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.support_vectors[:,0], clf.support_vectors[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(polynomial_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_soft():
        # X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        # X_train, y_train = split_train(X1, y1, X2, y2)
        # X_test, y_test = split_test(X1, y1, X2, y2)
        df = pd.read_csv('pulsar_star_dataset.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        X = X.to_numpy()
        y = y.to_numpy()
        y[y == 0] = -1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mean_train = X_train.mean()
        std_train = X_train.std()
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        # y_train.T.tofile('data1.txt', sep = '\n')
        # y_test.T.tofile('data2.txt', sep = '\n')
        # X_train.tofile('data3.txt', sep = '\n')
        # X_test.tofile('data4.txt', sep = '\n')
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        #pick 500 random points from train and test
        X_train = X_train[:800]
        y_train = y_train[:800]
        X_test = X_test[:200]
        y_test = y_test[:200]

        clf = SVM('gaussian', C=1.0)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        #print(y_predict)
        #y_predict = np.sign(y_predict)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        #plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
        
        
        #plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)

        
    #test_linear()
    #test_non_linear()
    test_soft()