import numpy as np
import random
import tqdm
class Predictor:
    def __init__(self,kernel, alpha=None, b=None, y=None, X=None, gama=None, w=None):
        self.alpha = alpha
        self.b = b
        self.kernel = kernel
        self.y = y
        self.X = X
        self.gama = gama
        if kernel == 'linear':
            self.w = w

    def predict(self, X_i):
        ans = 0
        if self.kernel == 'linear':
            ans = np.dot(self.w.T, X_i)
        elif self.kernel == 'rbf':
            dX = self.X - X_i
            ddx = np.multiply(dX, dX)
            ddx = np.sum(ddx, axis=1)
            ans = np.dot(np.exp(-self.gama * ddx).T, np.multiply(self.alpha, self.y))

        if ans + self.b > 0:
            return 1
        else:
            return -1



class SVM:
    def __init__(self, kernel, C):
        self.kernel = kernel
        self.c = C

    def train(self, X, y, max_iter):
        n = X.shape[0]
        self.n = X.shape[0]
        self.K = np.zeros((n,n))
        #print(self.K)
        k = 0
        self.b = 0
        alpha = np.zeros(n)
        self.w = np.dot(X.T, np.multiply(alpha,y))
        #SMO:
        while k < max_iter:
            temp_alpha = alpha.copy()
            for i in tqdm.tqdm(range(n-1)):
                alpha2_index = i
                alpha1_index = random.randrange(i+1, n)
                temp2 = self._get_K(alpha1_index, alpha1_index, X) + self._get_K(alpha2_index, alpha2_index, X)\
                        -2 * self._get_K(alpha1_index, alpha2_index, X)
                if self.KKT(i, self._get_F(i,X,alpha,y), y[i]) or temp2 == 0:
                    continue
                #计算α2的取值范围L,H
                L = 0
                H = self.c
                if y[alpha1_index] != y[alpha2_index]:
                    temp = alpha[alpha2_index] - alpha[alpha1_index]
                    L = max(0,temp)
                    H = min(self.c, self.c + temp)
                else:
                    temp = alpha[alpha2_index] + alpha[alpha1_index]
                    L = max(0, temp - self.c)
                    H = min(self.c, temp)
                #print("L:" + str(L) + " H:" + str(H))
                #求最优解α1，α2，b
                temp1 = y[alpha2_index]*(self._get_F(alpha1_index,X,alpha,y) - y[alpha1_index] -
                                         self._get_F(alpha2_index,X,alpha,y)  + y[alpha2_index])
                alpha2_star = alpha[alpha2_index] + temp1/temp2
                if alpha2_star > H:
                    alpha2_new_value = H
                elif alpha2_star < L:
                    alpha2_new_value = L
                else:
                    alpha2_new_value = alpha2_star
                alpha1_new_value = alpha[alpha1_index] + y[alpha1_index]*y[alpha2_index]*(alpha[alpha2_index]
                                                                                          -alpha2_new_value)
                alpha[alpha2_index] = alpha2_new_value
                alpha[alpha1_index] = alpha1_new_value
                self.w = np.dot(X.T, np.multiply(alpha,y))

                b1_star = y[alpha1_index] - (self._get_F(alpha1_index,X,alpha,y) - self.b)
                b2_star = y[alpha2_index] - (self._get_F(alpha2_index,X,alpha,y) - self.b)
                flag1 = alpha[alpha1_index] <= self.c
                flag2 = alpha[alpha2_index] <= self.c
                flag3 = alpha[alpha1_index] >= 0
                flag4 = alpha[alpha2_index] >= 0
                if flag1 and flag2 and flag3 and flag4:
                    self.b = b1_star
                else:
                    self.b = (b1_star + b2_star)/2

            minu = (temp_alpha-alpha)
            dst = np.dot(minu.T, minu)
            print(dst)
            if dst < 0.01:
                break
            k += 1
            print(k)
            if self.kernel == 'rbf':
                file = './rbf_alpha_' + str(k)
                np.save(file, alpha)
                file = './rbf_b_' + str(k)
                np.save(file, self.b)

        print("SMO has finished with k:", (k))
        return alpha, self.b

    def KKT(self, alpha_i_, fx_i, y_i):
        self.eplis = 0.001
        alpha_i = alpha_i_

        if alpha_i < 0 or alpha_i > self.c:
            return False
        elif alpha_i == 0 and fx_i*y_i >= 1:
            return True
        elif alpha_i == self.c and fx_i*y_i <= 1:
            return True
        elif fx_i*y_i == 1:
            return True
        return False

    def _get_K(self, i, j, X):
        gama = self.gama
        if self.K[i][j] == 0:
            if self.kernel == 'linear':
                self.K[i][j] = np.dot(X[i].T, X[j])
            elif self.kernel == 'rbf':
                self.K[i][j] = np.exp(-gama*np.linalg.norm((X[i]-X[j]), ord=2)**2)

        return self.K[i][j]

    def _get_F(self, i, X, alpha, y):
        temp = np.array(0)
        if self.kernel == 'linear':
            temp = np.dot(self.w.T, X[i])
        elif self.kernel == 'rbf':
            dX = X - X[i]
            ddx = np.multiply(dX, dX)
            ddx = np.sum(ddx, axis=1)
            temp = np.dot(np.exp(-self.gama*ddx).T, np.multiply(alpha, y))

        return temp+self.b

    def set_gama(self, gama):
        self.gama = gama

