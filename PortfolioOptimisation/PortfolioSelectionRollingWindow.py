import pandas as pd
from Tuning import *
from pandas import read_excel
def oneOrMinus(L):  # return -1 or 1 whichever is present max no. of time in List
    ones=0
    minus=0
    for i in L:
        if(i==1):
            ones+=1
        elif(i==-1):
            minus+=1
    if(ones >minus):
        return 1
    elif(minus>ones):
        return -1
    else:
        return 0



#df=read_excel(r"C:\Users\Ruchika\PycharmProjects\PortfolioOptimisation\PortfolioOptimisation\Nifty 50 (3).xlsx")
df=read_excel(r"C:\Users\Ruchika\Downloads\NIFTY50 2015-Present ALL.xlsx")
#sdf=df.iloc[:523,3:]
sdf=df.iloc[1:1861,3:]

selectdf=pd.DataFrame()
ind=["Trade High","Trade Low","Trade Close","Trade Volume"]
#get data of each asset
select=[]
#W=5
W=28 # no. of rolling
#D=50
D=180 #no. of training days on which rolling performed
#for i in range(0,20,4):
for i in range(0,192,4):
#for i in range(16, 200, 4):
    df=np.array(sdf.iloc[:,i+1:i+5])
    df=pd.DataFrame(df,columns=ind)
    selectrow=[]
    for j in range(W):
        dfw=df[j:D+j]
        r=Tuningandselecting(dfw,Target=0.002)[0]
        selectrow.append(r)

    print(selectrow)
    maxP=oneOrMinus(selectrow) #decide selected or not
    print(maxP)
    select.append(maxP)
    if maxP==1:
        selectdf[str(int(i/4))] = sdf.iloc[:,i+3:i+4]
    #print(select[int((i-16)/4)])
print(select)
df2 = pd.DataFrame(select)
print(df2)
print(selectdf)
df2.to_excel("shortlisted.xlsx", sheet_name='shortlist')
df2.to_excel("shortlisted_closingprices.xlsx", sheet_name='1')
# name=[]
# name_modified=[]
# for i in range(len(select)):
#     if(select[i]==1):
#         name_modified.append(name[i])

