import yfinance as yf
import numpy as np
from Optimization import *
import pandas as pd
from sklearn.decomposition import PCA
# Taking data of aaple and CAT
from Features import *
def Tuningandselecting(df,Target=0.005):
    PD,LP,HP,V= getPriceLowHighVolume(df)
    yd= ROC(PD,1) # rate return
    try:
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
    except:
        print("Zero division Error occured")
        return [0]
    # print(x7)
    # print("S.No.            y                Rate                       SMA               EMA                 ROC                MACD                   RSI               Will_R                  SO                AD                  MFI    ")
    # for i in range(len(x1)):
    #     print(i," ",yd[i]," ",x1[i],"  ",x2[i],"  ",x3[i],"  ",x4[i]," ",x5[i],"   ",x6[i],"  ",x7[i],"   ",x8[i],"   ",x9[i],"   ",x10[i])
    # X=[]

    #AFTER SHIFTING
    # print("AFTER SHIFTING")
    N=26
    x1=x1[N-1:len(x1)]
    x2=x2[N-1:len(x2)]
    x3=x3[N-1:len(x3)]
    x4=x4[N-1:len(x4)]
    x5=x5[N-1:len(x5)]
    x6=x6[N-1:len(x6)]
    x7=x7[N-1:len(x7)]
    x8=x8[N-1:len(x8)]
    x9=x9[N-1:len(x9)]
    x10=x10[N-1:len(x10)]
    yd=yd[N:]
    # print("SN,            y                Rate                       SMA               EMA                 ROC                 MACD                   RSI               Will_R                  SO                AD                  MFI      ")
    # for i in range(len(x1)-1):
    #     print(i," ",yd[i]," ",x1[i],"  ",x2[i],"  ",x3[i],"  ",x4[i]," ",x5[i],"   ",x6[i],"  ",x7[i],"  ",x8[i],"  ",x9[i],"  ",x10)
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
    # print("After Normalise")
    # print("SN            y                Rate                       SMA               EMA                 ROC                 MACD                   RSI               Will_R                  SO                AD                  MFI      ")
    #
    # for i in range(len(x1)-1):
    #     print(i," ",yd[i]," ",x1[i],"  ",x2[i],"  ",x3[i],"  ",x4[i]," ",x5[i],"   ",x6[i],"  ",x7[i],"  ",x8[i],"  ",x9[i],"  ",x10[i],)


    #making Feature Spce
    X=[]
    for i in range(len(x1)):
        X.append([x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i]])
    #changing y to +1,-1
    DF=pd.DataFrame(X,columns=["Rt","SMA" ,"EMA", "ROC", "MACD","RSI", "Will_R", "SO","AD" ,"MFI"])
    # print(DF)
    scaled_data = DF
    pca = PCA(0.95)  # create a PCA object
    pca.fit(scaled_data)  # do the math
    pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data
    X=np.array(pca_data)
    today = X[len(X) - 1]
    X=X[:len(X)-1]
    # print("todays data after PCA",today)
    y=[]
    for i in range(len(yd)):
        if yd[i] >= Target:
            y.append(1.0)
        else:
            y.append(-1.0)
    dat = np.array(X)
    labels = np.array(y)
    # pca = PCA(n_components=5)

    # #Do the Work
    C=[1,5,10,15,20]
    K=[0.2,0.4,0.6,0.8,0.9]
    FSigma=[0.1,0.2,0.4,0.6,0.8]
    Q=[0]
    E=100
    for i in C:
        for j in K:
            for fs in FSigma:
                #FS=SVM(dat,labels,C=i,Kernel='R',K_Var=j,Fsigma=None)
                FS=SVM(dat,labels,C=i,Kernel='R',K_Var=j,Fsigma=fs)
                FS.optimize_alpha()
                try:
                    Ker=FS.Kfold(3)
                except:
                    print("Zero division Error occured in K fold")
                    return [0]
                if Ker<E:
                    E=Ker
                    Q[0]=[i,j,fs]

   # print(Q,E)

    #check=SVM(dat,labels,C=1,Kernel='R',K_Var=0.2,Fsigma=0.1,Split_p=100)
    #check=SVM(dat,labels,C=1,Kernel='R',K_Var=0.2,Fsigma=None,Split_p=100)
    check=SVM(dat,labels,C=Q[0][0],Kernel='R',K_Var=Q[0][1],Fsigma=Q[0][2],Split_p=100)
    #check=SVM(dat,labels,C=Q[0][0],Kernel='R',K_Var=Q[0][1],Fsigma=None,Split_p=100)
    check.optimize_alpha()
    check.get_b()
    check.GetSupportVector()
    check.classify_NextDay(today)

    return check.classify_NextDay(today)


if __name__=="__main__":
    # df = yf.download(['AAPL'], start="2022-01-01", end="2022-06-01")
    # print(Tuningandselecting(df)[0])
    pass

