import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA 

#movie dataset 
cinema=pd.read_csv("tmdb_5000_movies.csv",encoding="latin-1")
cinema['genres']

#weighted average 
cinema['weight_avg']=cinema['vote_count']*(1/100)+cinema['vote_average']
cinema['weight_avg']
cinema['weight_avg'].quantile(0.90) 

cinema1=cinema[["id","budget","popularity","runtime","weight_avg"]].dropna() 
cinema2=cinema1.values 
cinema2.shape 
cinema1["weight_avg"].quantile(0.80)

#1. LDA (for feature scaling)
def weight_avg_class(x):
    if x>=16.1:
        return 1
    else:
        return 0

cinema1['class_score']=cinema1['weight_avg'].apply(weight_avg_class)
cinema1.head(5)
cinema1.class_score.unique() 

#data pre-process
X=cinema1.iloc[:,0:5]
y=cinema1.iloc[:,5]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)

#feature scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

#i. LDA
lda=LDA(n_components=1) #number of components= number of linear discriminants 
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)

#train and make predictions 
classifier=RandomForestClassifier(max_depth=2,random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

#evaluate the model performance
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred) #97.5% accuracy for classifying movie weighted rating?

#ii. PCA 
X=cinema1.drop('class_score',1)
y=cinema1['class_score']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.31) 

#normalize the data 
scaler=StandardScaler() 
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#PCA (only on the features not the labels)
pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

#explained variance ratio 
explained_var=pca.explained_variance_ratio_
explained_var

#train and make predictions
classifier=RandomForestClassifier(max_depth=2,random_state=0) 
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
cm=confusion_matrix(pred,y_test)
accuracy_score(pred,y_test) 
