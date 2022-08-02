from Tuning import *
from pandas import read_excel
df=read_excel(r"C:\Users\Ruchika\Downloads\Nifty 50 (3).xlsx")
sdf=df.iloc[:100,3:]
#get data of each asset
df1=sdf.iloc[:,1:5]
print(Tuningandsecting(df1,Target=0.002)[0])