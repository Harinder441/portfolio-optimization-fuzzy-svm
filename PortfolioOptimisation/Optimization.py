import numpy as np
import cvxopt
from Kernel import *
ZERO = 1e-5

class SVM:
    def __init__(self,X, y, Kernel='L', K_Var=None, C=1.0,Fsigma=None):
        self.X=X
        self.y=y
        self.Kernel=Kernel
        self.K_Var=K_Var
        self.C=C
        self.Fsigma=Fsigma
    def optimize_alpha(self):
        X,y,Kernel,K_Var,C,Fsigma=self.X,self.y,self.Kernel,self.K_Var,self.C,self.Fsigma
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = get_Ker(X[i], X[j],Kernel,K_Var)

        P = cvxopt.matrix(np.outer(y, y) * K)
        # w,v=eig(P)
        # print(W)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        print(A)
        b = cvxopt.matrix(0.0)

        if (Fsigma == None):
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        return a
    def GetSupportVector(self,a):
        sv = a > 1e-5
        self.SVind = np.arange(len(a))[sv]
        return len(self.SVind)
    def
if __name__=="__main__":
    X=np.array([[1,8],[4,5],[4,4],[1,1],[8,3]])
    y=np.array([1.0,1.0,1.0,1.0,-1.0])
    C=1
    kernel="L"
    sigma=None
    A=SVM(X,y)
    a=A.optimize_alpha()
    print(a)
