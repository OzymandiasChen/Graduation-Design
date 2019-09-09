
import numpy
import operator

'''
input:
    D:划分过的数据集
output:
    gini(D)：gini系数
'''
def calGini(D):
    DSize=len(D)
    C1=numpy.sum(D,axis=0)[-1]
    C0=DSize-C1
    return 1-(C1/DSize)*(C1/DSize)-(C0/DSize)*(C0/DSize)


'''
计算gini系数为离散化处理铺垫,gini系数越大与杂乱
Gini(D)=1-(Ck/D)^2
Gini(D,A)=D1/D*Gini(D1)+D2/D*Gini(D2)
input:
    dataSet:数据集
output:
    gini:gini阈值list
'''
def calGiniThreshold(dataSet):
    gini=[]
    attributionNum=len(dataSet[1])-1
    dataSize=len(dataSet)
    for attriIndex in range(attributionNum) :
        #print(attriIndex)
        sortedDataSet=dataSet[dataSet[:,attriIndex].argsort()]#分别按照每一个属性排序
        #对于第一个的计算ginithreshold,gini系数,算是第一个初始化,0,1划分
        bestGiniThreshold=(sortedDataSet[0,attriIndex]+sortedDataSet[1,attriIndex])/2.0
        giniD1=calGini(dataSet[0:1])
        giniD2=calGini(dataSet[1:dataSize])
        bestGiniD=1.0/dataSize*giniD1+(dataSize-1.0)/dataSize*giniD2
        #giniDA=
        dataIndex=2
        while dataIndex<dataSize :
            giniThreshold=(sortedDataSet[dataIndex-1,attriIndex]+sortedDataSet[dataIndex,attriIndex])/2.0
            giniD1=calGini(dataSet[0:dataIndex])
            giniD2=calGini(dataSet[dataIndex:dataSize])
            giniD=dataIndex/dataSize*giniD1+(dataSize-dataIndex)/dataSize*giniD2
            #更新
            if giniD<bestGiniD :
                bestGiniD=giniD
                bestGiniThreshold=giniThreshold
            dataIndex += 1
        gini.append(bestGiniThreshold)
    return gini

