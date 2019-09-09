import numpy
import random
from math import log,floor,ceil


'''
一个组件对应一个基分类器对应一个特征子集
clf的返回值是一个list
'''


'''
input:
    clfIndexList：装有基分类器索引的list
    baseClassifierSet：装有所有的基分类器
    attriIndexListSet：装有所有基分利器对应索引的list的list,200个属性集合
    data：idarray,一个数据
output:
    NT:正确分类的基分类器的比列
    NF:错误分类的基分类器的比列
    ensembleResult:当前集合的结果
'''
#集成分类器对于一条数据的结果
def ensembleClassify(clfIndexList,baseClassifierSet,attriIndexListSet,data):
    #一个组件对应一个基分类器对应一个特征子集
    clfNum=len(clfIndexList)
    lable=data[-1]   
    baseAns=[]
    for indexOfbaseClassfier in clfIndexList :#对于集合里的每一个组件
        testItem=[data[attriIndexListSet[indexOfbaseClassfier]].tolist()]
        testLable=baseClassifierSet[indexOfbaseClassfier].predict(testItem)
        baseAns.extend(testLable)
    if sum(baseAns)*2 >clfNum:
        ensembleResult=1
    elif sum(baseAns)*2 <clfNum:
        ensembleResult=0
    else:
        ensembleResult=random.randint(0,1)
    N1=sum(baseAns)/clfNum
    N0=1-N1
    if lable==1:
        NT=N1
        NF=N0
    else:
        NT=N0
        NF=N1
    return ensembleResult,NT,NF
  
      
'''
input:
    clfIndexList：装有基分类器索引的list
    baseClassifierSet：装有所有的基分类器
    attriIndexListSet：装有所有基分利器对应索引的list的list,200个属性集合
    baseEnsembleSet
    h：候选分类器索引位置
output:
    UWA:越大越好
'''
def calUWA(clfIndexList,baseClassifierSet,attriIndexListSet,h,dataSet) :#计算当前子集决策的不确定性的评价指标
    UWA=0
    for i in range(len(dataSet)) :
        lable=dataSet[i,-1]
        ensembleResult,NT,NF=ensembleClassify(clfIndexList,baseClassifierSet,attriIndexListSet,dataSet[i])
        testItem=[dataSet[i,attriIndexListSet[h]].tolist()]
        testLable=baseClassifierSet[h].predict(testItem)
        clfResult=testLable[0]
        if (clfResult==lable) & (ensembleResult==lable) : #ett
            UWA += NF
        elif (clfResult==lable) & (ensembleResult!=lable) :#etf
            UWA += NT
        elif (clfResult!=lable) & (ensembleResult==lable) :#eft
            UWA -= NF
        else :#eff
            UWA -= NT
    return UWA
        
  
    
def stopFlag(clfIndexList,baseClassifierSet,attriIndexListSet,dataSet) :
    dataSetSize=len(dataSet)
    cnt=0
    for i in range(dataSetSize) :
        ensembleResult,NT,NF=ensembleClassify(clfIndexList,baseClassifierSet,attriIndexListSet,dataSet[i])
        if ensembleResult == dataSet[i,-1] :
            cnt+=1
    if cnt == dataSetSize:
        return  1
    else :
        return 0



'''
基分类和最初的
input:
    dataSet:数据集合
    baseClassifierSet:基分类集合
    attriIndexListSet:包含属性索引的列表
    sizeOfEnsembleSet:一个集成分类器集合的大小
    rangeBeg:随机选择的始
    rangeEnd:随机选择的末
output:
    clfIndexList：随机初始化贪心选出的集合的分类器的位置的索引，list
'''
def RGSS(dataSet,baseClassifierSet,attriIndexListSet,sizeOfEnsembleSet,rangeBeg,rangeEnd) :#随机化贪心特征选择和投票
    ###最初的初始化
    clfIndexList=[]
    numOfBaseEnsembleSet=len(baseClassifierSet)
    clfMark=numpy.zeros(numOfBaseEnsembleSet)
    ###随机化的第一下
    clfIndex=random.randint(rangeBeg,rangeEnd)#随机选取第一个基分类器
    clfMark[clfIndex]=1
    clfIndexList.append(clfIndex)
    ####
    for i in range(sizeOfEnsembleSet-1) : #代表循环的数目
        #print('       ',i)
        if stopFlag(clfIndexList,baseClassifierSet,attriIndexListSet,dataSet) == 1 :
            break
        bestUWA=-100#基分类器越适合，UWA的值越大
        bestClfIndex=1#此时是无意义的值
        for k in range(numOfBaseEnsembleSet) : #代表穷举
            if clfMark[k]==0 :#对于没有入选的基分类器
                UWA=calUWA(clfIndexList,baseClassifierSet,attriIndexListSet,k,dataSet)
                if UWA >= bestUWA :
                    bestUWA=UWA
                    bestClfIndex=k
        clfMark[bestClfIndex]=1
        clfIndexList.append(bestClfIndex)
    return clfIndexList


'''
input:
    dataSet:数据集合
    baseClassifierSet:基分类集合
    attriIndexListSet:包含属性索引的列表的列表 2dlist
    sizeOfEnsembleSet:集成分类器集合的大小
    sizeOfComponentSet：组件网络的数目
output:
    componentSet：组件网络的集合 2dlist
'''

def chooseComponentSet(dataSet,baseClassifierSet,attriIndexListSet,sizeOfEnsembleSet,sizeOfComponentSet):
    baseClassifierSetSize=len(baseClassifierSet)
    step=floor(baseClassifierSetSize/sizeOfComponentSet)
    componentSet=[]
    for i in range(sizeOfComponentSet):
        #print(i)
        rangeBeg=i*step
        rangeEnd=(i+1)*step-1
        clfIndexList=RGSS(dataSet,baseClassifierSet,attriIndexListSet,sizeOfEnsembleSet,rangeBeg,rangeEnd)
        componentSet.append(clfIndexList)
    return componentSet


'''
input:
    attriIndexListSet:特征子集集合
    baseClassifierSet：特征子集对应的基分类器集合
    componentSet：选出来的组件集合的集合
    numOfFeatureSubSet:特征子集的数目，也是基分类器的数目，暂定200
    thresholdOfEnsemble：最终选入集成分类器的阈值
output:
    ensembleClassifier:最终的集成分类器
    ensembleIndexList:基分类器索引列表
    hashList:投票之后的hashlist，就是想看看结果
'''
def Ballot(attriIndexListSet,baseClassifierSet,componentSet,numOfFeatureSubSet,thresholdOfEnsemble):
    ensembleIndexList=[]
    ensembleClassifier=[]
    hashList=numpy.zeros(numOfFeatureSubSet)
    sumLenOfcomponentSetItem=0
    
    for componentSetItem in componentSet :
        sumLenOfcomponentSetItem+=len(componentSetItem)
        for index in componentSetItem:
            hashList[index] += 1
    '''
    aveNumOfBase=round(sumLenOfcomponentSetItem/len(componentSet))
    if (aveNumOfBase%2)==0:
        aveNumOfBase+=1
        
    if aveNumOfBase>1: 
        sizeOfEnsemble=aveNumOfBase-1
    #选出收敛大小个数的基分类器
    hashIndDescSort=hashList.argsort()[::-1][0:aveNumOfBase]
    
    for i in hashIndDescSort:
        ensembleIndexList.append(i)
        ensembleClassifier.append(baseClassifierSet[i])
    '''
    for i in range(numOfFeatureSubSet) :
        if hashList[i]>thresholdOfEnsemble:
            #print(i)
            ensembleIndexList.append(i)
            ensembleClassifier.append(baseClassifierSet[i])

    return ensembleClassifier,ensembleIndexList,hashList

'''
input:
    dataSet:数据集，2darray
    attriIndexListSet:特征子集集合
    baseClassifierSet：特征子集对应的基分类器集合
    sizeOfEnsembleSet:集成分类器集合的大小
    sizeOfComponentSet：组件网络的数目
    thresholdOfEnsemble：最终选入集成分类器的阈值
output:
    componentSet：组件网络的集合 2dlist
    ensembleClassifier:最终的集成分类器
    ensembleIndexList:基分类器索引列表
    ensemBasClfAtrInSet:ensemBasClfAtrInSet:最终的集成分对应的基分的属性的索引，2darray
    hashList:投票之后的hashlist，就是想看看结果
    aveNumOfBase:
''' 
def RGSSBEP(dataSet,baseClassifierSet,attriIndexListSet,sizeOfEnsembleSet,sizeOfComponentSet,thresholdOfEnsemble):
    componentSet=chooseComponentSet(dataSet,baseClassifierSet,attriIndexListSet,sizeOfEnsembleSet,sizeOfComponentSet)
    numOfFeatureSubSet=len(baseClassifierSet)
    ensembleClassifier,ensembleIndexList,hashList=Ballot(attriIndexListSet,baseClassifierSet,componentSet,numOfFeatureSubSet,thresholdOfEnsemble)
    ensemBasClfAtrInSet=numpy.array(attriIndexListSet)[ensembleIndexList]
    return componentSet,ensembleClassifier,ensemBasClfAtrInSet,ensembleIndexList,hashList
    
    
    