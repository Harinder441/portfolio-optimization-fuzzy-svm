import yfinance as yf
import numpy as np
from Features import *
from SVM_Fuzzy import *
import pandas as pd
from pandas import read_excel
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
# Taking data of aaple and CAT
df = yf.download(['DELL'],start="2022-01-01",end="2022-04-01")
# df=pd.read_excel(r"C:\Users\lenovo\Dropbox\PC\Documents\firstproject\DELL.xlsx")
date= df.index.to_list()
print(date)
with pd.ExcelWriter(r'C:\Users\lenovo\Dropbox\PC\Documents\firstproject\yahoo.xlsx',mode='a',if_sheet_exists="replace") as writer:
    df.to_excel(writer,sheet_name='yahoo')
PD,LP,HP,V= getPriceLowHighVolume(df)
yd= ROC(PD,1) # rate return
x1= ROC(PD,1) # rate return
x2=SMA(PD,5)
x3=EMA(PD,5)
x4=ROC(PD,3)
x5=MACD(PD)
x6=RSI(PD,14)
x7=Will_R(PD,HP,LP,14)
x8=SO(PD,HP,LP,14,3)
x9=AD(PD,HP,LP,V,14)
x10=MFI(PD,LP,HP,V,14)
# print(x7)
print("S.No.            y                Rate                       SMA               EMA                 ROC                MACD                   RSI               Will_R                  SO                AD                  MFI    ")
for i in range(len(x1)):
    print(i," ",yd[i]," ",x1[i],"  ",x2[i],"  ",x3[i],"  ",x4[i]," ",x5[i],"   ",x6[i],"  ",x7[i],"   ",x8[i],"   ",x9[i],"   ",x10[i])
X=[]

#AFTER SHIFTING
print("AFTER SHIFTING")
N=26
x1=x1[N-1:len(x1)-1]
x2=x2[N-1:len(x2)-1]
x3=x3[N-1:len(x3)-1]
x4=x4[N-1:len(x4)-1]
x5=x5[N-1:len(x5)-1]
x6=x6[N-1:len(x6)-1]
x7=x7[N-1:len(x7)-1]
x8=x8[N-1:len(x8)-1]
x9=x9[N-1:len(x9)-1]
x10=x10[N-1:len(x10)-1]
date=date[N:]
yd=yd[N:]
print("SN,            y                Rate                       SMA               EMA                 ROC                 MACD                   RSI               Will_R                  SO                AD                  MFI      ")
for i in range(len(x1)):
    print(i," ",yd[i]," ",x1[i],"  ",x2[i],"  ",x3[i],"  ",x4[i]," ",x5[i],"   ",x6[i],"  ",x7[i],"  ",x8[i],"  ",x9[i],"  ",x10)
x1=Normalise(x1) # rate return
x2=Normalise(x2)
x3=Normalise(x3)
x4=Normalise(x4)
x5=Normalise(x5)
x6=Normalise(x6)
x7=Normalise(x7)
x8=Normalise(x8)
x9=Normalise(x9)
x10=Normalise(x10)
print("After Normalise")
print("SN            y                Rate                       SMA               EMA                 ROC                 MACD                   RSI               Will_R                  SO                AD                  MFI      ")

#with pd.ExcelWriter(r'C:\Users\lenovo\Dropbox\PC\Documents\firstproject\x1.xlsx')as writer:
 #   x1.to_excel(writer,sheet_name='x1')

for i in range(len(x1)):
    print(i," ",yd[i]," ",x1[i],"  ",x2[i],"  ",x3[i],"  ",x4[i]," ",x5[i],"   ",x6[i],"  ",x7[i],"  ",x8[i],"  ",x9[i],"  ",x10[i],)


#making Feature Spce
X=[]
for i in range(len(x1)):
    X.append([x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i]])
#changing y to +1,-1
DF=pd.DataFrame(X,columns=["Rt","SMA" ,"EMA", "ROC", "MACD","RSI", "Will_R", "SO","AD" ,"MFI"],index=date)
print(DF)
with pd.ExcelWriter(r'C:\Users\lenovo\Dropbox\PC\Documents\firstproject\DF1.xlsx',mode='a',if_sheet_exists="replace")as writer:
     DF.to_excel(writer,sheet_name='DF1')
# First center and scale the data
scaled_data = DF

pca = PCA(0.95)  # create a PCA object
pca.fit(scaled_data)  # do the math
pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data

# The following code do constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

X=np.array(pca_data)
y=[]
for i in range(len(yd)):
    if yd[i] >= 0.005:
        y.append(1)
    else:
        y.append(-1)
print("X=",X)
print("y=",y)
dat = np.array(X)
labels = np.array(y)
# pca = PCA(n_components=5)
p=80
print("train=",(len(dat)/100)*p,"test=",(len(dat)/100)*(100-p))
F_sigma=0.2
kernel='R'
'''F_sigma,C,K_Var=0.1,0.1,0.1
while(C<1):
    F_sigma=0.1
    while(F_sigma<0.4):
        K_Var=0.1
        while(K_Var<1):
            print("sigma=",F_sigma,"C=",C,"K_Var=",K_Var)
            display_fuzzySVM_result(dat,labels,C,p,F_sigma,kernel,K_Var,M='FS')
            K_Var += 0.1
        F_sigma+=0.1
    C+=0.1
'''
K_Var=0.1
C=2
display_fuzzySVM_result(dat, labels, C, p, F_sigma, kernel, K_Var, M='FS')
# s=np.array([1 for i in range(len(dat[:4]))])
# s=get_siMember(dat[:4],F_sigma)
# alpha=optimize_alpha(dat[:4],labels[:4],kernel,None,C,s)
# print(alpha)
# b=get_w0(alpha,labels[:4],dat[:4],C,kernel,None)
# print(b)
# P=classify_points(dat[4:],dat[:4],labels[:4],alpha,b,kernel,None)
# print(P)

