import numpy as np
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor

class ContinuumArmedBandit:
    def __init__(self, X, y, convergence_rate=1.0):
        self.X = X
        self.y = y
        self.gpr = GPR(self.X, self.y, convergence_rate=convergence_rate)
        self.N = self.X.shape[0]

    def select_action(self):
        K = self.gpr.K
        noise_var = self.gpr.noise_var
        alpha = self.calc_alpha(K, noise_var, self.y)
        gamma = self.calc_gamma(K, noise_var, self.X)
        return (alpha, gamma)
    
    def calc_alpha(self, K, noise_var, y):
        alpha = np.linalg.inv((K + noise_var * np.eye(self.N))).dot(y)
        return alpha
    
    def calc_gamma(self, K, noise_var, X):
        beta = self.get_Beta(X)
        gamma = np.multiply(np.linalg.inv((K + noise_var * np.eye(self.N))), beta)
        return gamma
    
    def get_Beta(self, X):
        N = X.shape[0]
        Beta = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Beta[i,j] = self.beta(X[i], X[j])
        return Beta
    
    def beta(self, x1, x2):
        W = self.gpr.W
        beta_x1x2 = np.exp(-0.25 * (x1 - x2).T.dot(W).dot(x1 - x2))
        return beta_x1x2

class GPR(GaussianProcessRegressor):
    def __init__(self, X, y, convergence_rate=1.0):
        self.X = X
        self.y = y
        self.W = None
        self.K = None
        self.noise_var = None
        self.converge_rate_sq = np.power(convergence_rate, 2)
        GaussianProcessRegressor.__init__(self, kernel=self.get_kernel(self.X))
        self.fit(self.X, self.y)
        self.update_W()
        self.update_K(self.X)
        self.update_noise_var()
	
    def update(self, X, y):
        self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)
        self.fit(self.X, self.y)
        self.update_W()
        self.update_K(self.X)
        self.update_noise_var()
    
    def get_kernel(self, X):
        length_scale = np.random.normal(loc=1.0,scale=.1,size=X.shape[1])
        rbf = kernels.RBF(length_scale=length_scale)
        wk = kernels.WhiteKernel()
        kernel = rbf + wk
        return kernel

    def update_K(self, X):
        N = X.shape[0]
        K = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                K[i,j] = self.k(X[i], X[j])
        self.K = K

    def update_W(self):
        length_scales = self.kernel_.get_params()['k1__length_scale']
        w = 1.0 / np.power(length_scales, 2)
        W = np.diag(w)
        self.W = W

    def update_noise_var(self):
        noise_var = self.kernel_.get_params()['k2__noise_level']
        self.noise_var = noise_var


    def k(self, x1, x2):
        k_x1x2 = self.converge_rate_sq * np.exp(-0.5 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_x1x2

    def k_prime(self, x1, x2):
        k_prime_x1x2 = self.converge_rate_sq * np.exp(-1.0 * (x1 - x2).T.dot(self.W).dot(x1 - x2))
        return k_prime_x1x2
            
    
