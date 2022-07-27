import numpy as np
# For optimization
from scipy import linalg
import cvxopt
from sklearn.model_selection import GridSearchCV
from Kernels import *
ZERO = 1e-5


def optimize_alpha(X, y, Kernel, K_Var, C, s):
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

    if (C == None):
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
    else:
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

# def get_w(alpha, t, x):
#     m = len(x)
#     # Get all support vectors
#     w = np.zeros(x.shape[1])
#     for i in range(m):
#         w = w + alpha[i] * t[i] * x[i, :]
#     return w
def phi_x(x_test_i,x,t,alpha,kernel,K_Var): # x,y is the training data set
    ind_sv = np.where((alpha > ZERO))[0]
    s=0
    for i in ind_sv:
        s=s+alpha[i]*t[i]*get_Ker(x[i],x_test_i,kernel,K_Var)
    return s

def get_w0(alpha, t, x, C,kernel,K_Var):
    C_numeric = C - ZERO
    # Indices of support vectors with alpha<C
    ind_sv = np.where((alpha > ZERO) & (alpha < C_numeric))[0]
    w0 = 0.0
    for s in ind_sv:
        w0 = w0 + (-t[s] + phi_x(x[s],x,t,alpha,kernel,K_Var))
    # Take the average
    w0 = w0 / len(ind_sv)
    return w0

def classify_points(x_test,x,t,alpha,w0,kernel,K_Var):
    # get y(x_test)
    predicted_labels=[]
    for i in range(len(x_test)):
        predicted_labels.append(phi_x(x_test[i],x,t,alpha,kernel,K_Var)-w0)
    predicted_labels = np.array(predicted_labels)
    predicted_labels = np.sign(predicted_labels)
    # Assign a label arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels == ZERO] = 1
    return predicted_labels


def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors / total * 100
def train(x,t,p):
    x1=x[:int((len(x)/100)*p)]
    t=t[:int((len(x)/100)*p)]
    return x1,t
def test(x,t,p):
    x1=x[int((len(x)/100)*p):]
    t=t[int((len(x)/100)*p):]
    return x1,t
def get_siMember(x,sigma):
    l=len(x)
    t=[i for i in range(len(x)+1) ]
    s=[]
    for i in range(1,len(x)+1):
        si=((1-sigma)/(t[l]-t[1]))*t[i]+ (t[l]*sigma-t[1])/(t[l]-t[1])
        s.append(si)
    s=np.array(s)
    return s
def display_fuzzySVM_result(x, t, C,p,F_sigma,kernel,K_Var=None,M='FS'): #trainTestPercentage;M--->Model
    x_test,t_test=test(x,t,p)
    x_train,t_train=train(x,t,p)
    # Get the alphas
    # from sklearn.svm import SVC
    # classifier = SVC(kernel = 'rbf', random_state = 0,C=C,gamma=K_Var)
    # classifier.fit(x_train, t_train)
    # y_pred = classifier.predict(x_test)
    # print("Ypre",y_pred)
    # from sklearn.metrics import confusion_matrix, accuracy_score
    # cm = confusion_matrix(t_test, y_pred)
    # print(cm)
    # print(accuracy_score(t_test,y_pred))

    # if(M=='S'):
    #     s=np.array([1 for i in range(len(x_train))])
    # elif(M=='FS'):
    #     s=get_siMember(x_train,F_sigma)
    alpha = optimize_alpha(x_train, t_train,kernel,K_Var, C,s=None)
    print("alpha",alpha)
    # Get the weights
    # w = get_w(alpha, t_train, x_train)
    # print("weights",w)
    # w0 = get_w0(alpha, t_train, x_train, C,kernel,K_Var)
    # print("b",w0)
    # # Get the misclassification error and display it as title
    # predictions = classify_points(x_test,x_train,t_train,alpha,w0,kernel,K_Var)
    # print("prediction",predictions)
    # err = misclassification_rate(t_test, predictions)
    # print( 'C = ' + str(C) + ',  Errors: ' + '{:.1f}'.format(err) + '%')
    # print( ',  total SV = ' + str(len(alpha[alpha > ZERO])))

if __name__=="__main__":
    X=np.array([[1,8],[4,5],[4,4],[1,1],[8,3]])
    y=np.array([1.0,1.0,1.0,1.0,-1.0])
    C=1
    kernel="L"
    sigma=None
    a=optimize_alpha(X,y,kernel,sigma,C,None)
    print(a)
