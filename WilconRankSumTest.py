'''
Wilcoxcon rank sum test,WRST
'''
import scipy.stats
import numpy

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
'''
def selectAttriByRankSumTest(trainSet,purningSet,testSet,attributionNameList,attriNumAftRS) :
    PValueList=[]
    AttriNum=len(trainSet[0])-1
    for i in range(AttriNum) :
        PValue=RankSumTestAndCalPValue(trainSet[:,[i,-1]])
        PValueList.append(PValue)
    PValueList=numpy.array(PValueList)####1darray
    #PValueListRanked=
    PValueIndexList=PValueList.argsort()
    AttriINdexListAftRS=PValueIndexList[0:attriNumAftRS].tolist()
    
    attriNameARS=numpy.array(attributionNameList)[AttriINdexListAftRS]
    AttriINdexListAftRS.append(-1)
    trainSetARS=trainSet[:,AttriINdexListAftRS]
    purningSetARS=purningSet[:,AttriINdexListAftRS]
    testSetARS=testSet[:,AttriINdexListAftRS]
    '''
    i=0
    cnt=0
    while i<AttriNum:
        if PValueList[PValueIndexList][i]<0.000001:
            cnt+=1
        i+=1
    print(cnt)
    '''
    return trainSetARS,purningSetARS,testSetARS,attriNameARS


'''
对于每一个属性进行秩和检测，计算PValue
input:
    D:数据
output:
    PValue:P值
'''
def RankSumTestAndCalPValue(D) :
    RankIndexList=D[:,0].argsort().tolist()###list
    RankedList=numpy.array(D[D[:,0].argsort()])####array
    RankValueList=[]
    DSize=len(D)
    for i in range(DSize):
        RankValueList.append(i+1)
  
    pre=RankedList[0,0]
    beg=0#第一个重复值的起始位置
    cnt=1
    i=1
    while i<DSize:
        if RankedList[i,0]==pre:
            while (i<DSize) :
                if RankedList[i,0]==pre:
                    i+=1
                    cnt += 1
                else:
                    break;
            aveRank=sum(RankValueList[beg:beg+cnt])/cnt
            for k in range(beg,beg+cnt):
                RankValueList[k]=aveRank          
        else:
            pre=RankedList[i,0]
            beg=i
            cnt=1
            i+=1

    RankValueList=numpy.array(RankValueList)
    class0IndexInRankedList=[i for i,x in enumerate(RankedList[:,-1].tolist()) if x==0]
    class1IndexInRankedList=[i for i,x in enumerate(RankedList[:,-1].tolist()) if x==1]
    w0=sum(RankValueList[class0IndexInRankedList])
    w1=sum(RankValueList[class1IndexInRankedList])
    n0=len(class0IndexInRankedList)
    n1=len(class1IndexInRankedList)
    mu0=n0*(n0+n1+1)/2
    mu1=n1*(n0+n1+1)/2
    sigma=(n1*n0*(n0+n1+1)/2)**0.5
    z0=(w0-mu0)/sigma
    z1=(w1-mu1)/sigma
    #print('z0',z0,'z1',z1)
    PValue=2-scipy.stats.norm.cdf(abs(z0))-scipy.stats.norm.cdf(abs(z1))
    #print(PValue)
    return PValue
  
    
