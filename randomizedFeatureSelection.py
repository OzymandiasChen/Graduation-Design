'''
特征工程暂时选择24个特征
选择300个特征子集
'''

import numpy   #模块冲突 哎
import operator
import random
from math import log,floor


'''
input:
    D:D为装有2Darray的list
    featureIndex：用以划分地属性的索引
    giniThreshold：gini阈值
output：
    newD：划分之后的新的D
'''
def binSplitD(D,featureIndex,giniThreshold):
    newD=[]
    for arr in D:
        arr1=arr[arr[:,featureIndex]>giniThreshold]
        arr2=arr[arr[:,featureIndex]<=giniThreshold]
        if len(arr1) :
            newD.append(arr1)
        if len(arr2) :
            newD.append(arr2)  
    return newD 


'''
input:
    dataSet：数据集
output:
    shannonEnt:香农熵
'''
def calInfoGain(D,dataSize,featureIndex,giniThreshold,shannonEnt):
    newD=binSplitD(D,featureIndex,giniThreshold)
    newShannonEnt=calShannonEnt(newD,dataSize)
    infoGain=shannonEnt-newShannonEnt
    return infoGain
def calShannonEnt(D,dataSize):
    shannonEnt=0
    for arr in D :
        arrLen=len(arr)
        p1=sum(arr[:,-1])/arrLen
        p0=1-p1
        if p1!=0 :
            shannonEnt-=p1*log(p1,2)*arrLen/dataSize
        if p0!=0 :
            shannonEnt-=p0*log(p0,2)*arrLen/dataSize
    return shannonEnt


'''
randomize feature selection
备注：互信息作为启发信息
input:
    D:数据集
    gini:giniThreshold的list
    rangeBeg,rangeEnd:随机初始化的第一个属性的范围
output:
    attriIndexList:包含属性索引的列表
'''
def RFS(dataSet,gini,numOfFeature,rangeBeg,rangeEnd):   #randomize feature selection
    attributionNum=len(dataSet[0])-1
    dataSize=len(dataSet)
    D=[]#D为装有2Darray的list
    D.append(dataSet)
    mark=numpy.zeros(attributionNum)#没有被选择的特征对应位置为0，否则为1
    #选择第一个特征
    attriIndex=random.randint(rangeBeg,rangeEnd)#随机选取第一个特征，以后有待优化
    mark[attriIndex]=1
    D=binSplitD(D,attriIndex,gini[attriIndex])
    shannonEnt=calShannonEnt(D,dataSize)
    #bestAttri=attriIndex
    #okay
    for k in range(numOfFeature-1) :#再选择10个特征 
        bestInfoGain=0
        if shannonEnt==0 :#如果熵为0，退出
            break 
        for i in range(attributionNum) :
            if mark[i]==0 :
                infoGain=calInfoGain(D,dataSize,i,gini[i],shannonEnt)
                if infoGain>=bestInfoGain :
                    bestAttri=i
                    bestInfoGain=infoGain
        mark[bestAttri]=1
        D=binSplitD(D,bestAttri,gini[bestAttri])
        shannonEnt=calShannonEnt(D,dataSize)
    attriIndexList=[]
    for i in range(attributionNum) :
        if mark[i]==1 :
            attriIndexList.append(i)
    return attriIndexList

'''
numOfSubSet下随机化贪心特征选择
input:
    dataSet,gini
    numOfFeature:每个特征子集的大下，暂定25
    numOfSubSet:特征子集的数目，也是基分类器的数目，暂定300，
output:
    attriIndexListSet:属性索引的list的集合
'''
def choseFeatureSet(dataSet,gini,numOfFeature,numOfSubSet):
    attributionNum=len(dataSet[0])-1
    step=floor(attributionNum/numOfSubSet)
    attriIndexListSet=[]
    for i in range(numOfSubSet):
        #print(i)
        rangeBeg=i*step
        rangeEnd=(i+1)*step-1
        attriIndexList=RFS(dataSet,gini,numOfFeature,rangeBeg,rangeEnd)
        attriIndexListSet.append(attriIndexList)
    
    sumLen=0
    for i in range(numOfSubSet):
        sumLen+=len(attriIndexListSet[i])
        
    return attriIndexListSet,floor(sumLen/numOfSubSet)

