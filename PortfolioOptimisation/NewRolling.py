import pandas as pd
from Tuning import *
from pandas import read_excel
from openpyxl import load_workbook
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
df=read_excel(r"C:\Users\hp\Downloads\FTSE100ALL.xlsx",sheet_name='FilteredWithoutIndexAll')
#sdf=df.iloc[:523,3:]
sdf=df.iloc[1:1861,3:]
dfReturns=read_excel(r"C:\Users\hp\Downloads\FTSE100ALL.xlsx",sheet_name='ReturnsWithoutIndexAll')   #give path of return file here
sdfReturns=dfReturns.iloc[1:1861,3:]
selecteddataframes=[]
ind=["Trade High","Trade Low","Trade Close","Trade Volume"]
#get data of each asset
select=[]
#W=5
W=28 # no. of rolling
#D=50
D=180 #no. of training days on which rolling performed
#for i in range(0,20,4):
Add=60 # I think for SSD

for j in range(10,12):
    selectdf = pd.DataFrame()
    selectrow = []
    for i in range(0,376,4):
        df = np.array(sdf.iloc[:, i + 1:i + 5])
        df = pd.DataFrame(df, columns=ind)
        dfw = df[j+j*Add:D + j+j*Add]
        r = Tuningandselecting(dfw, Target=0.002)[0]
        selectrow.append(r)
        print(r)
        if r== 1:
            selectdf[str(int(i / 4))] = sdfReturns.iloc[j+j*Add:D + j+j*Add+60, i + 3:i + 4]
    # print(selectrow,selectdf)
    select.append(selectrow)
    selecteddataframes.append(selectdf)
print(select)
# df2 = pd.DataFrame(select)
# print(df2)
writein=pd.ExcelWriter("FTSE100SVM_10to11.xlsx", engine='xlsxwriter')
for i in range(len(selecteddataframes)):
    selecteddataframes[i].to_excel(writein, sheet_name='Rolling'+str(i))
writein.save()
# writein.close()
# df2.to_excel("shortlisted_closingprices.xlsx", sheet_name='1')
# name=[]
# name_modified=[]
# for i in range(len(select)):
#     if(select[i]==1):
#         name_modified.append(name[i])

