import pandas as pd
from Tuning import *
from pandas import read_excel
df=read_excel(r"C:\Users\Ruchika\PycharmProjects\PortfolioOptimisation\PortfolioOptimisation\Nifty 50 (3).xlsx")
sdf=df.iloc[:50,3:]
selectdf=pd.DataFrame()
ind=["Trade High","Trade Low","Trade Close","Trade Volume"]
#get data of each asset
select=[]
for i in range(0,40,4):
#for i in range(16, 200, 4):
    df=np.array(sdf.iloc[:,i+1:i+5])
    df=pd.DataFrame(df,columns=ind)
    r=Tuningandselecting(df,Target=0.002)[0]
    select.append(r)
    print(select[int(i/4)])
    if r==1:
        selectdf[str(int(i/4))] = sdf.iloc[:,i+3:i+4]

    #print(select[int((i-16)/4)])
print(select)
df2 = pd.DataFrame(select)
print(df2)
print(selectdf)
df2.to_excel("shortlisted.xlsx", sheet_name='shortlist')
# name=[]
# name_modified=[]
# for i in range(len(select)):
#     if(select[i]==1):
#         name_modified.append(name[i])

