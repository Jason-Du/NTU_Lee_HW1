import csv
import os
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd

'''
with open ('test.csv',newline='')as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		print(row)
'''

textTrain=open(os.path.join(os.path.dirname(__file__),"train.csv"),"r",encoding="big5")
textTest=open(os.path.join(os.path.dirname(__file__),"test.csv"),"r",encoding="big5")
rowTrain=csv.reader(textTrain)
rowTest=csv.reader(textTest)
listTrainData=[]
for i in range(18):
	listTrainData.append([])
n_row = 0
for r in rowTrain:
	if n_row != 0:
		for i in range(3, 27):#3~26
			if r[i] != "NR":
				listTrainData[(n_row-1) % 18].append(float(r[i]))
			else:
				listTrainData[(n_row-1) % 18].append(float(0))
	n_row += 1
textTrain.close()

listTestData=[]
n_row=0

for row in rowTest:
	if n_row%18==0:
		listTestData.append([])
	for i in range(2,11):
		if row[i]!="NR":
			listTestData[(n_row//18)].append(float(row[i]))
		else:
			listTestData[(n_row//18)].append(float(0))
	n_row += 1
textTest.close()

listTrainX = []
listTrainY = []
# 將資料拆成 x 和 y
for m in range(12):#0~11
    # 一個月每10小時算一筆資料，會有471筆
	#PM2.5
    for i in range(471):
        listTrainX.append([])
        listTrainY.append(listTrainData[9][480*m + i + 9])#一個月每10小時算一筆資料 一個月共有480
        # 18種汙染物
        for p in range(18):
        # 收集9小時的資料
            for t in range(9):
                listTrainX[471*m + i].append(listTrainData[p][480*m + i + t])
arrayTest=np.array(listTestData)
arrayTrainX=np.array(listTrainX)
arrayTrainY=np.array(listTrainY)
print(arrayTest.shape)
#TRAIN


# 增加bias項
# gradient decent

def GD(X,Y,W,etc,iteration,lamdba2):
	listcost=[]
	for itera in range(iteration):
		arraryY=X.dot(W)
		arrayloss=arraryY-Y
		arraycost=np.sum(arrayloss**2)/X.shape[0]
		listcost.append(arraycost)
		arraygradient=(X.T.dot(arrayloss)/X.shape[0])+(lamdba2*W)
		W-=etc*arraygradient
		if (itera%1000 == 0):
			print("iteraition:{},cost:{}".format(itera,arraycost))
	return W,listcost
def Adagrad(X,Y,W,etc,iteration,lamdba2):

	listcost=[]
	arraygradientsum=np.zeros(X.shape[1])
	for itera in range(iteration):
		arrayY=X.dot(W)
		arrayloss =arrayY-Y
		arraycost=np.sum(arrayloss**2)/X.shape[0]
		listcost.append(arraycost)
		arraygradient=X.T.dot(arrayloss)/X.shape[0]+lamdba2*W
		arraygradientsum+= arraygradient**2
		arraysigma=np.sqrt(arraygradientsum)
		W-=(etc*arraygradient)/arraysigma
		if iteration%10000==0:
			print("iterition:{},cost:{}".format(itera,arraycost))
	return W,listcost

arrayTrainX = np.concatenate((np.ones((arrayTrainX.shape[0], 1)), arrayTrainX), axis=1) # (5652, 163)
intLearningRate = 1e-6
arrayW = np.zeros(arrayTrainX.shape[1])
arrayW_G ,listcost_G=GD(X=arrayTrainX,Y=arrayTrainY,W=arrayW,etc=intLearningRate,iteration=20000,lamdba2=0)
intLearningRate=5
arrayW=np.zeros(arrayTrainX.shape[1])
arrayW_A,listcost_A=Adagrad(X=arrayTrainX,Y=arrayTrainY,iteration=20000,lamdba2=0,etc=intLearningRate,W=arrayW)
arrayW_C=inv(arrayTrainX.T.dot(arrayTrainX)).dot(arrayTrainX.T.dot(arrayTrainY))
print(len(listcost_A))

#TEST.......................................................................................................

arrayTest=np.concatenate((np.ones((arrayTest.shape[0],1)),arrayTest),axis=1)
arraypredict_gd=arrayTest.dot(arrayW_G)
arraypredict_ad=arrayTest.dot(arrayW_A)
arraypredict_cf=arrayTest.dot(arrayW_C)


#Drawing....................................................................................................
plt.plot(np.arange(len(listcost_G[3:])),listcost_G[3:],"b--",label="GD")
plt.plot(np.arange(len(listcost_A[3:])),listcost_A[3:],"r--",label="Adagrad")
plt.title("Train process")
plt.xlabel("iteration")
plt.ylabel("Cost  function")
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__),"trainprocess"))
plt.show()




dcit={"AD":arraypredict_ad,"CF":arraypredict_cf,"GD":arraypredict_gd}
pdresult= pd.DataFrame(dcit)
pdresult.to_csv(os.path.join(os.path.dirname(__file__),"predict"))
print(pdresult)






plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(np.arange(len(arraypredict_gd)),arraypredict_gd,"b--")
plt.title("GD")
plt.xlabel("testdataindex")
plt.ylabel("predict result")
plt.subplot(132)
plt.plot(np.arange(len(arraypredict_ad)),arraypredict_ad,"r--")
plt.title("AD")
plt.xlabel("testdataindex")
plt.ylabel("predict result")
plt.subplot(133)
plt.plot(np.arange(len(arraypredict_cf)),arraypredict_cf,"g--")
plt.title("CF")
plt.xlabel("testdataindex")
plt.ylabel("predict result")
plt.savefig(os.path.join(os.path.dirname(__file__), "Compare"))
plt.show()





'''



X=[1,2,3]
X=np.array(X)
print((3-X)**2)

arrayW = np.ones(X.shape)
print (X.dot(arrayW))

x=np.array([[1,2],[3,4],[5,6]])
print (x.shape)
print (x.T.dot([2,2,2]))


listTrainData=[]
for i in range(18):
	listTrainData.append([1])
print((listTrainData))

print(type(listTrainData))
print(np.array(listTrainData).shape)



print(arrayTest.shape)
print(
np.ones((4, 1))
)

x = np.array([[[1],[2]],[[3],[4]],[[5],[6]]]) #[1, 2, 3]
y = np.array([[[7],[8]],[[9],[10]],[[11],[12]]]) #[4, 5, 6]

x = np.array([[[1],[2],[10]],[[3],[4],[11]],[[5],[6],[12]]]) #[1, 2, 3]
y = np.array([[[7],[8],[13]],[[9],[10],[14]],[[11],[12],[15]]]) #[4, 5, 6]
print(
	x.ndim
)
print(
	x.shape
)
print(np.concatenate([x,y],axis=2))
'''
