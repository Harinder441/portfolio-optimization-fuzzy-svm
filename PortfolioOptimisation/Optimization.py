import numpy as np
import cvxopt
from Kernel import *
ZERO = 1e-5
from sklearn.model_selection import KFold
class SVM:
    def __init__(self,X, y, Kernel='L', K_Var=None, C=1.0,Fsigma=None,Split_p=80):
        self.X_train,self.y_train,self.X_test,self.y_test=self.SplitTrainTest(X,y,Split_p)
        self.X,self.y=X,y
        self.Kernel=Kernel
        self.K_Var=K_Var
        self.C=C
        self.Fsigma=Fsigma
        self.SVind=None
        self.b_=None
    def optimize_alpha(self,X=None,y=None):
        if type(X).__name__ == 'NoneType':
            X, y=self.X_train,self.y_train
        else:
            self.X_train, self.y_train=X,y
        Kernel,K_Var,C,Fsigma=self.Kernel,self.K_Var,self.C,self.Fsigma
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
        b = cvxopt.matrix(0.0)

        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        if (Fsigma == None):
            tmp2 = np.ones(n_samples) * C
        else:
            tmp2 = self.get_siMember() * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        cvxopt.solvers.options['show_progress']=False
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])
        self.alpha=alpha
        return self.alpha
    def GetSupportVector(self):
        alpha=self.alpha
        sv = alpha > 1e-5
        self.SVind = np.arange(len(alpha))[sv]
        return len(self.SVind)

    def phi_x(self,x_test_i,x,t,alpha):
        pass
    def get_b(self):
        alpha=self.alpha
        self.GetSupportVector()

        X,y=self.X_train,self.y_train
        C_numeric = self.C - ZERO
        # Indices of support vectors with alpha<C
        Cbound_sv_ind = np.where((alpha > ZERO) & (alpha < C_numeric))[0]
        b = []
        for j in Cbound_sv_ind:
            Sum=0
            for i in self.SVind:
                Sum=Sum+alpha[i]*y[i]*get_Ker(X[i], X[j],self.Kernel,self.K_Var)  #check something can done for getker
            b.append(-y[j] + Sum)
        print(b)
        # Take the average
        self.b_ = sum(b) / len(Cbound_sv_ind)
        return self.b_
    def classify_points(self):
        if self.b_==None:
            self.get_b()
        X,y=self.X_train,self.y_train
        predicted_labels = []
        for x in self.X_test:
            sum=0
            for i in self.SVind:
                sum+=self.alpha[i]*y[i]*get_Ker(X[i], x,self.Kernel,self.K_Var)
            predicted_labels.append(sum - self.b_)
        predicted_labels = np.array(predicted_labels)
        self.predicted_labels = np.sign(predicted_labels)
        # Assign a label arbitrarily a +1 if it is zero
        self.predicted_labels[predicted_labels == ZERO] = 1
        return self.predicted_labels

    def misclassification_rate(self,X=None,y=None):
        if type(X).__name__ != 'NoneType':
            self.X_test,self.y_test=X,y
        self.classify_points()
        total = len(self.y_test)
        errors = np.sum(self.y_test !=self.predicted_labels)
        return errors / total * 100

    def SplitTrainTest(self,X,y,Split_p):
        X_train = X[:int((len(X) / 100) * Split_p)]
        y_train = y[:int((len(X) / 100) * Split_p)]
        X_test = X[int((len(X) / 100) * Split_p):]
        y_test = y[int((len(X) / 100) * Split_p):]
        return X_train,y_train,X_test,y_test
    def get_siMember(self):
        l = len(self.X_train)
        t = [i for i in range(len(self.X_train) + 1)]
        s = []
        for i in range(1, len(self.X_train) + 1):
            si = ((1 - self.Fsigma) / (t[l] - t[1])) * t[i] + (t[l] * self.Fsigma - t[1]) / (t[l] - t[1])
            s.append(si)
        s = np.array(s)
        return s
    def Kfold(self,n_splits=3):
        kf = KFold(n_splits=n_splits)
        A=0
        for tr_i, ts_i in kf.split(self.X):
            self.optimize_alpha(self.X[tr_i],self.y[tr_i])
            A+=self.misclassification_rate(self.X[ts_i],self.y[ts_i])
        return A/n_splits
if __name__=="__main__":
    X=np.array([[1,8],[4,5],[4,4],[1,1],[8,3]])
    y=np.array([1.0,1.0,1.0,1.0,-1.0])
    C=1
    kernel="L"
    sigma=None
    A=SVM(X,y,Split_p=100)
    a=A.optimize_alpha()
    b=A.get_b(a)
    # E=A.misclassification_rate(a)
    print(a,b)
#https://mbhaskar1.github.io/machine%20learning/2019/07/04/svm-using-cvxopt.html
#https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download