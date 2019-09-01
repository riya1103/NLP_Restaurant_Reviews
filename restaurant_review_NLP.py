import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords1=stopwords.words('english')
stopwords1.sort()
stopwords1.remove('no')
stopwords1.remove('not')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if  not word  in set(stopwords1)]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 500)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
#PCA didn't improve accuracy!

from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=250)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.metrics import average_precision_score, recall_score
average_precision = average_precision_score(y_test, y_pred)
recall=recall_score(y_test,y_pred)






















































