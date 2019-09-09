import numpy



'''
attriIndexList:包含属性索引的列表,2d List
'''
def atrriCount(attriIndexListSet,attriNameARS):
    cnt=0
    attributionNum=len(attriNameARS)
    hashList=hashList=numpy.zeros(attributionNum)
    for attriIndexList in attriIndexListSet :
        for index in attriIndexList :
            hashList[index] +=1
    for i in range(attributionNum):
        if hashList[i]>5 :
            cnt+=1
            print(i,':',hashList[i],attriNameARS[i])
    print('attriCnt:',cnt)
    
    