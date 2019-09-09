
'''
from imp import reload
import os,sys
os.chdir('C:\\Users\\Administrator\\Desktop\\bysj\\project')
sys.path.append('C:\\Users\\Administrator\\Desktop\\bysj\\project')
'''


'''
import main
#filename='ArabidopsisDrought.arff'
#filename='ArabidopsisNitrogen.arff'
filename='ArabidopsisTEV.arff'#readinfo有细节要改
precisionList,recallList,F1List,accList,P,R,F1,acc,aveSizeOfAtrriList=main.crossValidation(filename)
'''

'''
十折交叉验证，六折训练集，两折剪枝集，两折测试集
'''
import numpy
import readInfo
import WilconRankSumTest
import calGini
import randomizedFeatureSelection
import trainBaseClassifiers
import RGSSBEP
import test


'''
input:
    filename:文件名称
output:
    五折交叉验证后的
    acc：精度，预测正确的比例
    precision：查准率，预测的正例结果中真正的正例所占的比例
    recall：查全率，在所有的真实的正例中，有多大比例的正例被真正的预测出来了
    F1：precision和recall的调和平均数   
    precisionList:每一折的查准率记录
    recallList：每一折的召回率记录
    F1List：每一折的F1记录
    accList：每一折的精度记录
    aveSizeOfAtrriList:平均每一次选择的属性的数目的list
    aveSizeOfConpoSetList：每一次集成分类器中组件的数目的list
'''

def crossValidation(filename):####filename='ArabidopsisDrought.arff'
    '''
    读出数据以及切分
    attributionList:属性名称列表，list
    classList:类别名称列表，list
    dataSet:数据集 2darray
    '''
    k=10#十折交叉验证的10
    attributionNameList,classList,dataSet=readInfo.loadData(filename)
    step=round(len(dataSet)/k)
    D=[]
    for i in range(k-1):
        Di=dataSet[i*step:(i+1)*step,:]
        D.append(Di)
    D.append(dataSet[(k-1)*step:-1,:])
    
    '''
    数据的组合，以及每一折交叉检验
        0:训练集里，1：剪枝集里，2：测试集里
    '''

    mark=[[1,1,2,2,0,0,0,0,0,0],
          [0,1,1,2,2,0,0,0,0,0],
          [0,0,1,1,2,2,0,0,0,0],
          [0,0,0,1,1,2,2,0,0,0],
          [0,0,0,0,1,1,2,2,0,0],
          [0,0,0,0,0,1,1,2,2,0],
          [0,0,0,0,0,0,1,1,2,2],
          [2,0,0,0,0,0,0,1,1,2],
          [2,2,0,0,0,0,0,0,1,1],
          [1,2,2,0,0,0,0,0,0,1]]
    '''
    mark=[[1,2,0,0,0],
          [0,1,2,0,0],
          [0,0,1,2,0],
          [0,0,0,1,2],
          [2,0,0,0,1]]
    '''
    precisionList=[]
    recallList=[]
    F1List=[]
    accList=[]
    aveSizeOfAtrriList=[]
    #aveSizeOfConpoSetList=[]
    
    for Cross in range(k):#对于每一折
        print(Cross,':')
        trainList=[]
        purningList=[]
        testList=[]
        markItem=mark[Cross]
        for i in range(k):
            if markItem[i]==0:
                trainList.append(D[i])
            if markItem[i]==1:
                purningList.append(D[i])
            if markItem[i]==2:
                testList.append(D[i])
                
        trainSet=numpy.concatenate(tuple(trainList),axis=0)
        purningSet=numpy.concatenate(tuple(purningList),axis=0)
        testSet=numpy.concatenate(tuple(testList),axis=0)     
        precision,recall,acc,F1,aveSizeOfAtrri=RFSAndRGSSBEP(trainSet,purningSet,testSet,attributionNameList)    
        
        precisionList.append(precision)
        recallList.append(recall)
        F1List.append(F1)
        accList.append(acc)
        aveSizeOfAtrriList.append(aveSizeOfAtrri)
        #aveSizeOfConpoSetList.append(aveSizeOfConpoSet)
        
    P=sum(precisionList)/k
    R=sum(recallList)/k
    F1Ave=sum(F1List)/k
    accAve=sum(accList)/k
        
    print('Finally... ...')
    print('Pave:',P,'RAve:',R,'F1Ave:',F1Ave,'accAve:',accAve)
        
    return precisionList,recallList,F1List,accList,P,R,F1Ave,accAve,aveSizeOfAtrriList

        
            
'''
input:
    trainSet:训练集 2darray
    purningSet:剪枝集合，剪枝过程使用，2darray
    testSet：测试集 2darray
    attributionList:属性名称列表，list
output:
    precision:最终的预测结果中，真正的正例所占的比例
    recall:正例当中有多大比例被识别了
    acc:准确率,有多少被精准地预测了
    F1:准确率和召回率的调和平均数，precission和recall是矛盾的
'''  

def RFSAndRGSSBEP(trainSet,purningSet,testSet,attributionNameList):
    '''
    秩和检测降维，因为基因的维度实在太大，普通的台式机实在跑不动，十几天
    input:
        trainSet:2darray
        attriNumAftRS:秩和检测所选出的基因的数目
        purningSet:剪枝集合，剪枝过程使用，2darray
        testSet:测试集 2darray
        attributionNameList:属性名称列表，list
    output:
        trainSetARS:秩和检测处理后的训练集2darray
        purningSetARS:秩和检测处理后剪枝集合，剪枝过程使用，2darray
        testSetARS:秩和检测处理后的测试集2darray
        attriNameARS:秩和检测处理过的属性名称1darray
        aveSizeOfAtrri:平均每一次选择的属性的数目
        aveSizeOfConpoSet：每一次集成分类器中组件的数目
    '''
    #print('wilconson Rank Sum Test')
    attriNumAftRS=1000
    trainSetARS,purningSetARS,testSetARS,attriNameARS=WilconRankSumTest.selectAttriByRankSumTest(trainSet,purningSet,testSet,attributionNameList,attriNumAftRS)
    
    '''
    计算gini系数为离散化处理铺垫,gini系数越大与杂乱
    input:
        trainSetARS:训练集，秩和检测处理后的训练集2darray
    output:
        gini:gini阈值list
    '''
    #print('calculate gini')
    gini=calGini.calGiniThreshold(trainSetARS)
    
    '''
    numOfSubSet下随机化贪心特征选择
    input:
        dataSet,gini
        numOfFeatureSubSet:每个特征子集的大下，暂定25
        numOfSubSet:特征子集的数目，也是基分类器的数目，暂定200，
    output:
        attriIndexListSet:属性索引的list的集合，这个index是基于attriNameARS的
        aveSizeOfAtrri:
    '''
    #print('choose attri')
    numOfFeature=25#numOfFeature:每个特征子集的大下，暂定25
    numOfFeatureSubSet=200#numOfSubSet:特征子集的数目，也是基分类器的数目，暂定200
    attriIndexListSet,aveSizeOfAtrri=randomizedFeatureSelection.choseFeatureSet(trainSetARS,gini,numOfFeature,numOfFeatureSubSet)
    
    '''
    统计各个属性被选入的频率，一个可视化的效果
    import attriCnt
    attriCnt.atrriCount(attriIndexListSet,attriNameARS)
    '''
    
    '''
    input:
        trainSetARS:训练集，秩和检测处理后的训练集2darray
        attriIndexListSet:属性索引的list的集合，这个index是基于attriNameARS的list
    output:
        baseClassifierSet:根据200个特征子集所训练出来基分利器,list
        
    '''
    #print('train base')
    baseClassifierSet=trainBaseClassifiers.svmTrain(trainSetARS,attriIndexListSet)
    
    '''
    input:
        purningSetARS:秩和检测处理后剪枝集合，剪枝过程使用，2darray
        baseClassifierSet:根据200个特征子集所训练出来基分利器,list
        attriIndexListSet:属性索引的list的集合，这个index是基于attriNameARS的list
        sizeOfEnsembleSet:每次随机选择一个组件集合的分类器的最多的数目
        sizeOfClassifierComponentSet:总共多少次随机选择
        thresholdOfEnsemble：最终选入集成分类器的票数阈值
    output:
        ensembleClassifier：最终的集成分类器 list
        ensembleIndexList：最终的集成分类器的索引 list,可以对应特征子集的索引
        componentSet：组件网络索引的list的list
        aveSizeOfConpoSet:
        ensemBasClfAtrInSet:最终的集成分对应的基分的属性的索引，2darray
    '''
    #print('choose base')
    numOfClassifierComponentSet=30#组件集合的数目
    sizeOfEnsembleSet=30#组件集合的大小
    thresholdOfEnsemble=10
    componentSet,ensembleClassifier,ensemBasClfAtrInSet,ensembleIndexList,hashList=RGSSBEP.RGSSBEP(purningSetARS,baseClassifierSet,attriIndexListSet,sizeOfEnsembleSet,numOfClassifierComponentSet,thresholdOfEnsemble)
    # componentSet,ensembleClassifier,ensemBasClfAtrInSet,ensembleIndexList,hashList
    '''
    input:
        ensembleClassifier:最终的集成分类器 list
        ensemBasClfAtrInSet:最终的集成分对应的基分的属性的索引，2darray
        testSetARS:测试集 2darray
    output:
        precision:最终的预测结果中，真正的正例所占的比例
        recall:正例当中有多大比例被识别了
        acc:准确率,有多少被精准地预测了
        F1:准确率和召回率的调和平均数，precission和recall是矛盾的
    '''
    precision,recall,acc,F1=test.test(ensembleClassifier,ensemBasClfAtrInSet,testSetARS)
    return precision,recall,acc,F1,aveSizeOfAtrri

