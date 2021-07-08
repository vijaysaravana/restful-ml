from joblib import dump, load                                                      
import numpy                                                                       
from sklearn import svm                                                            
from sklearn import datasets                                                       
                                                                                                                                                                   
def train():                                                                       
    clf = svm.SVC()                                                                
    X, y = datasets.load_iris(return_X_y=True)                                     
    clf.fit(X, y)                                                                  
    dump(clf, 'iris.joblib')                                                                                                                                          
                                                                                   
def predict(data):                                                                 
    clf = load('iris.joblib')                                                      
    data = numpy.array(data).reshape(1, -1)                                        
    prediction = clf.predict(data)                                                 
    return {"labels": prediction.tolist()}   