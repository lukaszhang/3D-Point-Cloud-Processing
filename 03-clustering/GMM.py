# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')
#import KMeans

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        GMM.Mu = []
        GMM.Var = []
        GMM.pi = []
        GMM.gamma = []

    
    # 屏蔽开始
    # 更新W
    def update_gamma(self, data, Mu, Var, pi):
        pdf = np.zeros((data.shape[0], self.n_clusters), dtype=np.float64)
        for i in range(self.n_clusters):
            pdf[:, i] = pi[i] * multivariate_normal.pdf(data, Mu[i], Var[i])
        gamma = pdf / (pdf.sum(axis=1).reshape(-1,1))
        return gamma, pdf

    # 更新pi
    def update_pi(self, N, N_k):
        pi = N_k / N
        return pi
        
    # 更新Mu
    def update_Mu(self, N_k, data, gamma):
        N, dim = data.shape
        Mu = np.zeros((self.n_clusters, dim))
        for i in range(self.n_clusters):
            Mu[i] = np.sum(data * gamma[:,i].reshape(-1,1), axis=0) / N_k[i]
        return Mu

    # 更新Var
    def update_Var(self, data, N_k, gamma, Mu):
        #N, dim = data.shape
        Var = np.zeros_like(GMM.Var)
        for i in range(self.n_clusters):
            Var[i] = np.dot((data - Mu[i]).T, np.dot(np.diag(gamma.T[i].ravel()), data-Mu[i])) / N_k[i]
        return Var

    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # initialize parameters
        N = data.shape[0]
        GMM.pi = np.asarray(1 / self.n_clusters).repeat(self.n_clusters)
        #kmeans = KMeans.K_Means(n_clusters=self.n_clusters)
        #kmeans.fit(data)
        #GMM.Mu = kmeans.center
        #print(GMM.Mu)
        GMM.Mu = data[np.random.choice(np.arange(N), size=self.n_clusters, replace=False)]
        #GMM.Var = np.expand_dims([[1., 0.1], [0.2, 1.]], axis=0).repeat(self.n_clusters, axis=0)
        GMM.Var = np.asarray([np.cov(data, rowvar=False)] * self.n_clusters)
        last_log_likelihood_sum = 0
        iter_num = 0

        for i in range(self.max_iter):
            GMM.gamma, pdf = GMM.update_gamma(self, data, GMM.Mu, GMM.Var, GMM.pi)
            N_k = np.sum(GMM.gamma, axis=0)
            GMM.pi = GMM.update_pi(self, N, N_k)
            GMM.Mu = GMM.update_Mu(self, N_k, data, GMM.gamma)
            GMM.Var = GMM.update_Var(self, data, N_k, GMM.gamma, GMM.Mu)

            log_likelihood_sum = np.sum(np.log(pdf.sum(axis=1).reshape(-1,1)))
            #print(np.log(pdf.sum(axis=1).reshape(-1,1)))
            if np.abs(log_likelihood_sum-last_log_likelihood_sum) < 1:
                break
            last_log_likelihood_sum = log_likelihood_sum
            iter_num += 1
        #print(iter_num)
        #print(N_k)
        #print(GMM.pi)
        #print(GMM.Mu)
        #print(GMM.Var)
        #print(GMM.gamma)
        #print(GMM.gamma.shape)

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        #GMM.gamma, pdf = GMM.update_gamma(self, data, GMM.Mu, GMM.Var, GMM.pi)
        result = np.argmax(GMM.gamma, axis=1)
        #result = label.T
        #result = np.zeros((data.shape[0], 1), dtype=np.int_)
        #for i in range(data.shape[0]):
        #    result[i] = np.argmax(GMM.gamma[i, :])
        return result
        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

