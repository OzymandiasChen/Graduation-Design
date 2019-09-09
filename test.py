
import numpy

'''
input:
    ensembleClassifier:训练好的集成分类器对应的列表 list
    ensemBasClfAtrInSet:集成中的基分类器对应的属性列表 2darray
    data:一条数据,1darray
'''
def ensembleClassify(ensembleClassifier,ensemBasClfAtrInSet,data):
    baseClassifierNum=len(ensembleClassifier)
    baseAns=[]
    for i in range(baseClassifierNum):
        testItem=[data[ensemBasClfAtrInSet[i]].tolist()]
        testLable=ensembleClassifier[i].predict(testItem)
        baseAns.extend(testLable)
    if sum(baseAns)*2 >=baseClassifierNum:
        ensembleResult=1
    else:
        ensembleResult=0
    return ensembleResult    

'''
input:
    ensembleClassifier：最终的集成分类器 list
    ensembleIndexList：最终的集成分类器的索引 list,可以对应特征子集的索引
    attriIndexListSet:属性索引的list的集合，这个index是基于attriNameARS的list
    testSetARS:秩和检测处理后的测试集2darray
output:
'''
def test(ensembleClassifier,ensemBasClfAtrInSet,testSetARS):
    testSetSize=len(testSetARS)
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(testSetSize):
        label=testSetARS[i,-1]
        testLabel=ensembleClassify(ensembleClassifier,ensemBasClfAtrInSet,testSetARS[i,:])
        if (testLabel==1) &(label==1):
            TP+=1
        if (testLabel==0) &(label==1) :
            FN += 1
        if (testLabel==1) &(label==0) :
            FP+=1
        if (testLabel==0) &(label==0) :
            TN+=1
    print('TP',TP,'FN',FN,'FP',FP,'TN',TN)
    if (TP+FP)==0:
        precision=0.00001
    else:
        precision=TP/(TP+FP)
    if (TP+FN)==0:
        recall=0.99999
    else:
        recall=TP/(TP+FN)
    acc=(TP+TN)/(TP+TN+FP+FN)
    if (recall==0)or(precision==0):
        F1=0.00001
    else:
        F1=1/(0.5*(1/precision+1/recall))
    print('acc:',acc,'P:',precision,'R:',recall,'F1:',F1)
    return precision,recall,acc,F1
    
    
    
    
    
    
    
    