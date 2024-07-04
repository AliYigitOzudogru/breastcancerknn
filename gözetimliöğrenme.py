# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:14:24 2024

@author: Ali Yiğit Özüdoğru
"""


#the first algorithm is KNN algorithm 

# adımlar
# 1) Veri seti incelenmesi
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data,columns=cancer.feature_names)
df["Sonuç"] = cancer.target



#model aşaması KNN modeli seçildi
X= cancer.data # değerler
y=cancer.target # sonuç

#train test split/ayrılması

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# ölçeklendirme 

scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)


knn = KNeighborsClassifier()
knn.fit(x_train,y_train) # fit fonksiyonu modelimizi knn algoritması kullanarak eğitir

# sonuçların değerlendirilmesi
y_predict = knn.predict(x_test)
acc=accuracy_score(y_test, y_predict) #accuracy value
print("accuracy value is:",acc)
