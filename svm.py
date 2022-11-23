import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('/Users/himanish/Documents/LASTRIDE.xlsx')

np.random.shuffle(dataset.values)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0) #i need type casting 
dataset.dtypes


classifier.fit(X_train, y_train)  

y_pred= classifier.predict(X_test)  
classifier.fit(X_train, y_train)  


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

import pickle
filename = 'dt_svmmodel.sav'
pickle.dump(classifier, open(filename, 'wb'))

filename = 'dt_svmmodel.sav'

DT = pickle.load (open(filename, 'rb'))
#filename.fit(X,y)

result =classifier.score(X_test,y_test)
print(result)