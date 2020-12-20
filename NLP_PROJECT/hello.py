# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:49:21 2020

@author: Salman
"""
import numpy as np ## scientific computation
import pandas as pd ## loading dataset file
import matplotlib.pyplot as plt ## Visulization
import nltk  ## Preprocessing Reviews
nltk.download('stopwords') ##Downloading stopwords
from nltk.corpus import stopwords ## removing all the stop words
from nltk.stem.porter import PorterStemmer ## stemming of words
import re  ## To use Regular expression

dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t",quoting=3)




corpus = []
for i in range(0,1000):   #we have 1000 reviews
     review = re.sub('[^a-zA-Z]'," ",dataset["Review"][i])
     review = review.lower()
     review = review.split()
     pe = PorterStemmer()
     all_stopword = stopwords.words('english')
     all_stopword.remove('not')
     review = [pe.stem(word) for word in review if not word in set(all_stopword)]
     review = " ".join(review)
     corpus.append(review)
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) ##1500 columns
X = cv.fit_transform(corpus).toarray()
y = dataset["Liked"]

import pickle
pickle.dump(cv, open('cv.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB,MultinomialNB
GNB = GaussianNB()
MNB = MultinomialNB()

GNB.fit(X_train, y_train)
MNB.fit(X_train, y_train)

print(GNB.score(X_test,y_test))   ## 0.73
print(MNB.score(X_test,y_test))   ## 0.775

y_pred=MNB.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test,y_pred)
print(cm,score*100)

import pickle
pickle.dump(cv, open('cv.pkl', 'wb'))

loaded_model = pickle.load(open("cv.pkl", "rb"))
#y_pred_new = loaded_model.predict(X_test)
y_pred_new=MNB.predict(X_test)
#loaded_model.score(X_test,y_test)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred_new)
score = accuracy_score(y_test,y_pred_new)
print(cm,score*100)

def new_review(new_review):
  new_review = new_review
  new_review = re.sub('[^a-zA-Z]', ' ', new_review)
  new_review = new_review.lower()
  new_review = new_review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  new_review = [ps.stem(word) for word in new_review if not word in  set(all_stopwords)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  new_X_test = cv.transform(new_corpus).toarray()
  #print(new_X_test.shape)
  new_y_pred = MNB.predict(new_X_test)
  return new_y_pred
new_review = new_review(str(input("Enter new review...")))
if new_review[0]==1:
   print("Positive")
else :
   print("Negative")





