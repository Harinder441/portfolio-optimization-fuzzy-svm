import pandas as pd
from Tuning import *
from pandas import read_excel
df=read_excel(r"C:\Users\Ruchika\PycharmProjects\PortfolioOptimisation\PortfolioOptimisation\Nifty 50 (3).xlsx")
sdf=df.iloc[:523,3:]

ind=["Trade High","Trade Low","Trade Close","Trade Volume"]
#get data of each asset
select=[]
for i in range(0,200,4):
    df=np.array(sdf.iloc[:,i+1:i+5])
    df=pd.DataFrame(df,columns=ind)
    select.append(Tuningandselecting(df,Target=0.002)[0])
    print(select[int(i/4)])
print(select)

