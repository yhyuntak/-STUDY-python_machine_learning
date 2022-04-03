import pandas as pd

review_df = pd.read_csv("word2vec-nlp-tutorial/labeledTrainData.tsv",header=0,sep="\t",quoting=3)
print(review_df.head(3))
import re
review_df['review']=review_df['review'].str.replace('<br />',' ')
review_df['review'] = review_df['review'].apply(lambda x : re.sub("[^a-zA-Z]"," ",x))
#
# from sklearn.model_selection import train_test_split
# class_df = review_df['sentiment']
# feature_df = review_df.drop(['id','sentiment'],axis=1,inplace=False)
# X_train,X_test,y_train,y_test = train_test_split(feature_df,class_df,test_size=0.3,random_state=156)
#
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score,roc_auc_score
#
# pipeline = Pipeline([('cnt_vect',CountVectorizer(stop_words='english',ngram_range=(1,2))),
#                      ('lr_clf',LogisticRegression(C=10))])
#
# pipeline.fit(X_train['review'],y_train)
# pred = pipeline.predict(X_test['review'])
# pred_probs = pipeline.predict_proba(X_test['review'])[:,1]
#
# print("CountVectorizer의 예측 정확도 : {0:.4f}, ROC-AUC : {1:.4f}".format(accuracy_score(y_test,pred),roc_auc_score(y_test,pred_probs)))
#
# pipeline = Pipeline([('tfidf_vect',TfidfVectorizer(stop_words='english',ngram_range=(1,2))),
#                      ('lr_clf',LogisticRegression(C=10))])
#
# pipeline.fit(X_train['review'],y_train)
# pred = pipeline.predict(X_test['review'])
# pred_probs = pipeline.predict_proba(X_test['review'])[:,1]
#
# print("TfidfVectorizer의 예측 정확도 : {0:.4f}, ROC-AUC : {1:.4f}".format(accuracy_score(y_test,pred),roc_auc_score(y_test,pred_probs)))

import nltk
nltk.download('all')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_anlayzer = SentimentIntensityAnalyzer()
senti_scores = senti_anlayzer.polarity_scores(train_df['review'][0])
print(senti_scores)

# 감성분석 뒷부분은 안함. 읽기만