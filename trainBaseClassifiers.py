from sklearn import svm

def svmTrain(dataSet,attriIndexList) :
    svmList=[]
    for attriList in attriIndexList :
        trainSet=dataSet[:,attriList].tolist()
        trainLableSet=dataSet[:,-1].tolist()
        clf = svm.SVC()  # class
        clf.fit(trainSet,trainLableSet)
        svmList.append(clf)
    return svmList
        

    
    '''
    attriIndexList=[]
    for i in range(len(mark)):
        if mark[i]==1:
            attriIndexList.append(i)
    trainSet=dataSet[0:45,attriIndexList].tolist()
    testSet=dataSet[45:59,attriIndexList].tolist()
    trainLable=dataSet[0:45,-1].tolist()
    testLable=dataSet[45:59,-1].tolist()
    clf = svm.SVC()  # class   
    clf.fit(trainSet,trainLable)
    for i in range(len(testSet)) :
        predictLable=clf.predict(testSet[i])
        print('predictLable=',predictLable,'realLabl=',testLable[i])
    return trainSet,testSet,trainLable,testLable,clf
    '''
    


'''
from sklearn import svm  
  
X = [[0, 0], [1, 1], [1, 0]]  # training samples   
y = [0, 1, 1]  # training target  
clf = svm.SVC()  # class   
clf.fit(X, y)  # training the svc model  
  
result = clf.predict([2, 2]) # predict the target of testing samples   
print result  # target   
  
print clf.support_vectors_  #support vectors  
  
print clf.support_  # indeices of support vectors  
  
print clf.n_support_  # number of support vectors for each class  
''' 