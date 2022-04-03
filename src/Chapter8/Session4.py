from sklearn.datasets import fetch_20newsgroups
news_data = fetch_20newsgroups(subset='all',random_state=156)

print(news_data.keys())

import pandas as pd

print("target 클래스의 값과 분포도 ",pd.Series(news_data.target).value_counts().sort_index())
print("target 클래스의 이름들 ",news_data.target_names)


train_news = fetch_20newsgroups(subset='train',remove=('headers','footers','quotes'),random_state=156)
X_train = train_news.data
y_train = train_news.target

test_news = fetch_20newsgroups(subset='test',remove=('headers','footers','quotes'),random_state=156)
X_test = test_news.data
y_test = test_news.target

# from sklearn.feature_extraction.text import CountVectorizer
# cnt_vect = CountVectorizer()
# cnt_vect.fit(X_train)
# X_train_cnt_vect = cnt_vect.transform(X_train)
# X_test_cnt_vect = cnt_vect.transform(X_test)
#
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#
# lr_clf = LogisticRegression()
# lr_clf.fit(X_train_cnt_vect,y_train)
# pred = lr_clf.predict(X_test_cnt_vect)
# print("CountVEctorized LOgistic Regression의 예측 정확도는 {0:.3f}".format(accuracy_score(y_test,pred)))


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_df=300)
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)
print(type(X_test_tfidf_vect))
#
# lr_clf = LogisticRegression()
# lr_clf.fit(X_train_tfidf_vect,y_train)
# pred = lr_clf.predict(X_test_tfidf_vect)
# print("TfidfVectorized LOgistic Regression의 예측 정확도는 {0:.3f}".format(accuracy_score(y_test,pred)))
