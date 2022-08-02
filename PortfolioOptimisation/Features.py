# D-- Data list
# PD-- Price data
import numpy as np
def EMA(D, N): #ith day included
    P_EMA = D.copy()
    k = 2/(N+1)
    E_0 = sum(D[:N])/N
    for i in range(N):
        P_EMA[i] = 0
    P_EMA[N-1] = E_0
    for i in range(N, len(D)):
         P_EMA[i] = D[i]*k+P_EMA[i-1]*(1-k)
    return P_EMA
def SMA(D,N): #ith  day  included SMA at day i= pi
    P_SMA = D.copy()
    for i in range(N):
        P_SMA[i] = 0
    for i in range(N-1, len(D)):
        P_SMA[i] = sum(D[i+1-N:i+1])/N
    return P_SMA
def ROC(PD,N): # x_t= {P_t-P_(t-N)}/P_(t-N)
     X= PD.copy()
     for i in range(N):
        X[i] = 0
     for t in range(N,len(PD)):
         X[t]=(PD[t]-PD[t-N])/PD[t-N]  #t_th day
     return X
def MACD(PD): # Ema12-Ema 26
    if(len(PD)<=26):
        print("ERROR: MACD can't be calculated")
        return PD
    else:
        E12=EMA(PD,12)
        E26=EMA(PD,26)
        M=PD.copy()
        for i in range(26-1): # -1 bcz indx start from 0
            M[i]=0
        for t in range(26-1,len(PD)):
            M[t]=E12[t]-E26[t]
        return M
def RSI(PD,N):
    Gain=[] # 0 bc 1st day NAN
    Loss=[]
    for i in range(1,len(PD)):
        change=PD[i]-PD[i-1]
        if(change>=0):
            Gain.append(change)
            Loss.append(0)
        else:
            Gain.append(0)
            Loss.append(-change)

    AvgG=SMA(Gain,N)
    AvgL=SMA(Loss,N)
    AvgL.insert(0,0)  # 0 bc 1st day NAN
    AvgG.insert(0,0)
    # print("G",Gain,"\n","L",Loss,"\n","AL",AvgL,"\n","AG",AvgG)
    RSI=PD.copy()
    for i in range(N):
        RSI[i] = 0
    for t in range(N,len(PD)):
        RSI[t]=100-(100/(1+(AvgG[t]/AvgL[t])))
    return RSI
def Will_R(PD,HP,LP,N): #PD-- closing price
    HHP=[0 for i in range(len(PD))]
    LLP=[0 for i in range(len(PD))]
    Will=[0 for i in range(len(PD))]
    for i in range(N-1,len(PD)):
        HHP[i]=max(HP[i+1-N:i+1])
        LLP[i]=min(LP[i+1-N:i+1])
        Will[i]=((PD[i]-HHP[i])/(HHP[i]-LLP[i]))*100
    # print(HHP,"\n",LLP)
    return Will
def SO(PD,HP,LP,N,N1):
    HHP=[0 for i in range(len(PD))]
    LLP=[0 for i in range(len(PD))]
    SO=[0 for i in range(len(PD))]
    for i in range(N-1,len(PD)):
        HHP[i]=max(HP[i+1-N:i+1])
        LLP[i]=min(LP[i+1-N:i+1])
        SO[i]=((PD[i]-LLP[i])/(HHP[i]-LLP[i]))*100
    # print(HHP,"\n",LLP,"\n",SO)
    SO_SMA=SMA(SO[N-1:],N1)
    for i in range(N-1):
        SO_SMA.insert(i,0)
    return SO_SMA
def AD(PD,HP,LP,V,N):
    HHP=[0 for i in range(len(PD))]
    LLP=[0 for i in range(len(PD))]
    AD=[0 for i in range(len(PD))]
    for i in range(N-1,len(PD)):
        HHP[i]=max(HP[i+1-N:i+1])
        LLP[i]=min(LP[i+1-N:i+1])
        AD[i]=((2*PD[i]-LLP[i]-HHP[i])/(HHP[i]-LLP[i]))*V[i]
    return AD
def ADX(n):
    pass
def MFR(CD,LP,HP,V,N):
    RMF=[]
    TP=[]
    PRMF=[]
    NRMF=[]
    MFR=[0 for i in range(len(CD))]
    for i in range(len(CD)):
        TP.append((CD[i]+LP[i]+HP[i])/3)
        RMF.append(TP[i]*V[i])
        change=TP[i]-TP[i-1]
        if(i==0):
            PRMF.append(0)
            NRMF.append(0)
        elif(change>=0):
            PRMF.append(RMF[i])
            NRMF.append(0)
        else:
            NRMF.append(RMF[i])
            PRMF.append(0)
    for t in range(N,len(CD)):
        MFR[t]=(sum(PRMF[t+1-N:t+1])/sum(NRMF[t+1-N:t+1]))
    return MFR


def MFI(CD,LP,HP,V,N):
    MFRL=MFR(CD,LP,HP,V,N)
    MFI=[0 for i in range(len(CD))]
    for t in range(N,len(CD)):
        MFI[t]=100-(100/(1+MFRL[t]))
    return MFI
def selectFeature(W):
    pass
def getFeatureSpace(PD,High=None,Low=None,Volume=None,Features=["R1","SMA","EMA","ROC","MACD","RSI",]): #"WILL","SON","AD"
    pass
def getPriceLowHighVolume(df):
    return list(df['Trade Close']),list(df['Trade High']),list(df['Trade Low']),list(df['Trade Volume'])
def Normalise(D):
    X=D.copy()
    X_max=max(X)
    X_min= min(X)
    for i in  range(len(X)):
        X[i]= (X[i]-X_min)/(X_max-X_min)
    return X
if __name__=="__main__":
    '''L=np.random.rand(20)
    print(L)
    N=5
    High=[
    5,
    6,
    5,
    5,
    6,
    6,
    6,
    6,
    9,
    6,
    6,
    6,
    5,
    5,
    8,
    55,
    53,
    52,
    50,
    53,
    58]
    Low=[56,
57,
5,
5,
57,
58,
59,
9,
90,
5,
59,
57,
57,
55,
53,
53,
54,
55,
54,
55
]
    Volume=[2906500,
5745700,
6049000,
3751300,
3516200,
10511600,
7313500,
3098100,
4133000,
5790400,
5053900,
3957500,
2893700,
4212100,
5523700,
4421900,
3126700,
4359300,
3920500,
3576800]'''

 # S=SMA(L,N)
 # print(S)

#print(RSI(L,5))
    # print(ROC(L,2))