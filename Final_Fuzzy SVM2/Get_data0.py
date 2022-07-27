import yfinance as yf
import numpy as np
from Features import *
from SVM_Fuzzy import *
# Taking data of aaple and CAT
df = yf.download(['RELIANCE.NS'],start="2022-01-01",end="2022-04-01")
print(len(df),df,df['Close'])
# daily percentage Return

print(df)
df=df.iloc[1:]

x1=list(df)
x3=EMA(x1,5)
N=5
M=1
x2=SMA(x1,5)
print(x2)

R=x1[N-1:len(x1)-M] # M is the no. days


X=[]
for i in range(len(R)):
    X.append([R[i],x2[i],x3[i]])

y=[]
for i in range(N,len(x1)):
    if x1[i] >= 0.001:
        y.append(1)
    else:
        y.append(-1)
print("X=",X)
print("y=",y)
dat = np.array(X)
labels = np.array(y)
C=981
p=80
print("train=",(len(dat)/100)*p,"test=",(len(dat)/100)*(100-p))
F_sigma=0.1
kernel='R'
display_fuzzySVM_result(dat,labels,C,p,F_sigma,kernel,K_Var=0.7,M='FS')
# s=np.array([1 for i in range(len(dat[:4]))])
# s=get_siMember(dat[:4],F_sigma)
# alpha=optimize_alpha(dat[:4],labels[:4],kernel,None,C,s)
# print(alpha)
# b=get_w0(alpha,labels[:4],dat[:4],C,kernel,None)
# print(b)
# P=classify_points(dat[4:],dat[:4],labels[:4],alpha,b,kernel,None)
# print(P)

