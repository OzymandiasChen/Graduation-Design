

import numpy   #模块冲突 哎
import operator


'''
读取数据，
input:
    filename：文件名称，需要带后缀
output:
    attributionList：属性名字对应的list,
    classList：类别名字list,
    dataSet：数据监测np.array,最后一列为lable
'''
def loadData(filename):
    fr=open(filename)
    arffFile=fr.readlines() #读取文件
    index=1
    lenOfArffFile=len(arffFile)
    #
    attributionList=[]
    #classList=['wet','dry']#  wet为0，dry为1#ArabidopsisDrought.arff
    #classList=['KNO3','KCL']#  KNO3为0，KCL为1#ArabidopsisNitrogen.arff,后边有一个对应的地方要改
    classList=['infected','non-infected']#  infected为0，non-infected为1#ArabidopsisTEV.arff,后边有一个对应的地方要改
    #
    #读取属性
    while (arffFile[index][0:10]=="@ATTRIBUTE") :
        attributionList.append(arffFile[index][0:-6])
        index += 1
     
    attributionList=attributionList[0:-1]
    index-=1
    
    index +=2
    dataSet=numpy.zeros((lenOfArffFile-index,len(attributionList)+1))
    indexOfData=0
    while index < lenOfArffFile :
        line=arffFile[index].strip()
        line=line.split(',')
        newLine=line[0:-1]
        #if line[-1]=='wet' :
        if line[-1]=='infected' :
        #if line[-1]=='infected' :
            newLine.append(0)
            #print(0)
        else :
            newLine.append(1)
            #print(1)
        dataSet[indexOfData,:]=newLine
        index += 1
        indexOfData += 1
    #标准化S  改成计算gini系数
    #dataMax=numpy.max(dataSet,axis=0)
    #dataMin=numpy.min(dataSet,axis=0)
    #chazhi=dataMax-dataMin
    #dataSet=(dataSet-dataMin)/chazhi
    #
    fr.close()
    return attributionList,classList,dataSet